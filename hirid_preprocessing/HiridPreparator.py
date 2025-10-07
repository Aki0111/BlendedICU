from pathlib import Path
from string import printable
from typing import Type

import pandas as pd
import polars as pl

from database_processing.datapreparator import DataPreparator
from database_processing.newmedicationprocessor import NewMedicationProcessor

class hiridPreparator(DataPreparator):
    def __init__(
            self,
            variable_ref_path,
            ts_path,
            pharma_path,
            admissions_path,
            imputedstage_path):
    # variable_ref_path='reference_data/hirid_variable_reference.csv',
    # ts_path='observation_tables/parquet/',
    # pharma_path='pharma_records/parquet/',
    # admissions_path='reference_data/general_table.csv',
    # imputedstage_path='imputed_stage/parquet/'
        super().__init__(dataset='hirid', col_stayid='admissionid')
        self.col_los = 'lengthofstay'
        self.unit_los = 'second'
        self.variable_ref_path = self.source_pth + variable_ref_path
        #self.variable_ref_path = E:/Science/database/hirid/reference_data/hirid_variable_reference.csv
        self.admissions_path = self.source_pth + admissions_path
        self.ts_path = self.source_pth + ts_path
        self.med_path = self.source_pth + pharma_path
        #self.med_path = E:/Science/database/hirid/pharma_records/parquet/
        self.imputedstage_path = self.source_pth + imputedstage_path
        
        self.variable_ref_parquet_path = self.raw_as_parquet_pth + self._get_name_as_parquet(self.variable_ref_path)
        self.admissions_parquet_path = self.raw_as_parquet_pth + self._get_name_as_parquet(self.admissions_path)
        self.ts_parquet_path = self.raw_as_parquet_pth + self._get_name_as_parquet(Path(self.ts_path).parent)
        self.med_parquet_path = self.raw_as_parquet_pth + self._get_name_as_parquet(Path(self.med_path).parent)
        #_get_name_as_parquet=pharma_records.parquet
        self.imputedstage_parquet_path = self.raw_as_parquet_pth + self._get_name_as_parquet(Path(self.imputedstage_path).parent)
        #self.raw_as_parquet_pth = E:/Science/Blended/hirid_data/raw_parquet/
        # self.med_parquet_path = E:/Science/Blended/hirid_data/raw_parquet/pharma_records.parquet
        # 构造各个数据文件对应的Parquet格式文件的存储路径
        # 通过将原始文件路径转换为Parquet文件名，并拼接Parquet文件的基础存储目录，
        # 生成各个数据表对应的Parquet格式文件的完整存储路径
        self._check_files_untarred()
        
        self.ts_savepth = self.savepath + 'timeseries.parquet'
        #self.savepath = E:/Science/Blended/eicu_data/
        self.weights = None
        self.heights = None
        

    def raw_tables_to_parquet(self):
        """
        Writes initial csv.gz files to parquet files. This operations 
        needs only to be done once and allows further methods to be 
        done laziy using polars.
        """
        pths_as_parquet = {
                self.variable_ref_path: (False, ','),
                self.admissions_path: (False, ','),
                self.imputedstage_path: (True, None),
                self.ts_path: (True, None),
                self.med_path: (True, None),
                }
        #pth_as_parquet = 
        for i, (src_pth, (src_is_multiple_parquet, sep)) in enumerate(pths_as_parquet.items()):
            
            if src_is_multiple_parquet:
                tgt = self.raw_as_parquet_pth + self._get_name_as_parquet(Path(src_pth).parent)
            else:
                tgt = self.raw_as_parquet_pth + self._get_name_as_parquet(src_pth)
            
            if Path(tgt).is_file() and i==0:
                print(f'{tgt} already exists, skipping conversion to parquet.i={i}')
                inp = input('Some parquet files already exist, skip conversion to parquet ?[n], y')
                if not inp:  # 如果输入为空
                    inp = 'y'
                if inp.lower() == 'y':
                    break
            print(f'写入：Converting {tgt} to parquet...')
            # 写入文件,parquet转换
            if i==0:
                self.write_as_parquet(src_pth,
                                      tgt,
                                      astype_dic={},
                                      encoding='utf-8-sig',
                                      sep=sep,
                                      src_is_multiple_parquet=src_is_multiple_parquet)
            else:
                self.write_as_parquet(src_pth,
                    tgt,
                    astype_dic={},
                    encoding='unicode_escape',
                    sep=sep,
                    src_is_multiple_parquet=src_is_multiple_parquet)
           
    def init_gen(self):
        self.id_mapping = self._variablenames_mapping()
        self.lazyadmissions, self.admissions = self._load_admissions()
        
    def _check_files_untarred(self):
        '''Checks that files were properly untarred at step 0.'''
        notfound = False
        files = [self.ts_path,
                 self.med_path,
                 self.imputedstage_path,
                 self.admissions_path]
        for file in files:
            if not Path(file).exists():
                notfound = True
                print(f'{file} was not found, consider running step 0 to'
                      ' untar Hirid source files\n')
        if notfound:
            raise ValueError('Some files are missing, see warnings above.')
    
    def _load_los(self):
        """
        As is usually done with this database, the length of stay is defined as
        the last timeseries measurement of a patient.
        """
        timeseries = pl.scan_parquet(self.imputedstage_parquet_path)

        los = (timeseries
               .select('patientid', 'reldatetime')
               .drop_nulls()
               .rename({'patientid': 'admissionid',
                        'reldatetime': 'lengthofstay'})
               .group_by('admissionid')
               .max()
               .with_columns(
                   pl.duration(seconds=pl.col('lengthofstay')).alias('lengthofstay')
                   ))
                
        return los
        
    
    def _load_admissions(self):
        try:
            adm = pl.scan_parquet(self.admissions_parquet_path)
        except FileNotFoundError:
            print(self.admissions_parquet_path,
                  'was not found.\n run raw_tables_to_parquet first.' )
            return None, None

        adm = (adm
               .rename({'patientid': 'admissionid'})
               .with_columns(
                   admissiontime=pl.col('admissiontime').str.to_datetime(),
                   admissionid=pl.col('admissionid').cast(pl.Int32)
                   )
               )
        return adm, adm.collect().to_pandas()

    def _variablenames_mapping(self):
        try:        
            lf = pl.scan_parquet(self.variable_ref_parquet_path)
            #E:/Science/Blended/hirid_data/raw_parquet/hirid_variable_reference.parquet
        except FileNotFoundError:
            print(self.variable_ref_parquet_path,
                  'was not found.\n run raw_tables_to_parquet first.' )
            return None
        variable_ref = (lf
                        .select('Source Table',
                                'ID',
                                'Variable Name')
                        .collect()
                        .to_pandas()
                        .dropna()
                        )
#          Source Table       ID                        Variable Name
# 0    Observation      200                           Heart rate
# 1    Observation      410                Core body temperature
# 2    Observation     7100                   Rectal temperature
# 3    Observation      400                 Axillary temperature
# 4    Observation      100  Invasive systolic arterial pressure
# ..           ...      ...                                  ...
# 707       Pharma  1001045                   Madopar Tbl 125 mg
# 708       Pharma  1000266                       PK-Merz 500 ml
# 709       Pharma  1000406               Madopar LIQ 125 mg Tbl
# 710       Pharma  1000620                    Neupogen 48 Mio U
# 711       Pharma  1000619                    Neupogen 30 Mio U
        idx_obs = variable_ref['Source Table'] == 'Observation'
        idx_pharma = variable_ref['Source Table'] == 'Pharma'

        obs_id_mapping = variable_ref.loc[idx_obs]
        pharma_id_mapping = variable_ref.loc[idx_pharma]

        obs_id_mapping = dict(zip(obs_id_mapping['ID'],
                                  obs_id_mapping['Variable Name']))

        pharma_id_mapping = dict(zip(pharma_id_mapping['ID'],
                                     pharma_id_mapping['Variable Name']))

        obs_id_mapping[310] = 'Respiratory rate id=310'
        obs_id_mapping[5685] = 'Respiratory rate id=5685'
        # print(f'调试obsid={obs_id_mapping}')
        # print(f'调试pharma={pharma_id_mapping}')
        return {'observation': obs_id_mapping,
                'pharma': pharma_id_mapping}

    def _load_heights_weights(self):
        print('Fetching heights and weights in timeseries data, this step '
              'takes several minutes.')
        variables = {'weight': 10000400,
                     'height': 10000450}
        
        ts = pl.scan_parquet(self.ts_parquet_path)
        # E:/Science/Blended/hirid_data/raw_parquet/observation_tables.parquet 
        # ttttt=self.lazyadmissions.collect()
        # zzzz=ts.collect()
        # print(f'调试ts_parquet_pth={self.ts_parquet_path},ts={zzzz}')
        # print(f'调试self.lazyadmissions={ttttt},ts={zzzz}')

        df = (ts
             .select('datetime',
                     'patientid',
                     'value',
                     'variableid')
             .rename({'datetime': 'valuedate',
                      'patientid': 'admissionid'})
             .filter((pl.col('variableid')==variables['height']) 
                     | (pl.col('variableid')==variables['weight']))
             .join(self.lazyadmissions.select('admissiontime', 'admissionid'),
                   on='admissionid')
             .with_columns(
                 valuedate = pl.col('valuedate') - pl.col('admissiontime')
                 )
             .filter(pl.col('valuedate')<pl.duration(seconds=self.flat_hr_from_adm.total_seconds()))
             .drop('valuedate', 'admissiontime')
             .group_by('admissionid', 'variableid')
             .mean()
             .collect(streaming=True))
        
        partitions = df.partition_by(['variableid'], as_dict=True)
        
        weights = (partitions[variables['weight'],]
                           .rename({'value': 'weight'})
                           .drop('variableid')
                           .lazy())
        
        heights = (partitions[variables['height'],]
                           .rename({'value': 'height'})
                           .drop('variableid')
                           .lazy())
        
        print('  -> Done')
        return heights, weights

    def gen_labels(self):
        """
        The admission table does not contain the heights and weights. 
        These variables must be fetched from the timeseries table.
        The length of stay (los) is not specified either.
        It is usually derived from the last measurement of a timeseries
        variable. 
        """
        print('o Labels')
        
        lengthsofstay = self._load_los()

        if (self.heights is None) or (self.weights is None):
            self.heights, self.weights = self._load_heights_weights()
        admissions = (self.lazyadmissions
                      .join(self.heights, on='admissionid', how='left')
                      .join(self.weights, on='admissionid', how='left')
                      .join(lengthsofstay, on='admissionid', how='left')
                      .with_columns(
                          care_site=pl.lit('Bern University Hospital')
                          ))

        self.save(admissions, self.savepath+'labels.parquet')

    def gen_timeseries(self):
        self.labels = pl.scan_parquet(self.labels_savepath)
		
        self.get_labels(lazy=True)
        # print(f'调试self.lablers={self.labels}')
        # print(f'调试self.stays={self.stays}')
        ##self.stays = self.labels.select(self.col_stayid).unique().collect().to_numpy()住院天数
        ts = pl.scan_parquet(self.ts_parquet_path)
        # str_ob_mapping = {str(k): v for k, v in self.id_mapping['pharma'].items()}
        lf = (ts
              .select(['datetime', 'patientid', 'value', 'variableid'])
              .with_columns(pl.col('patientid').alias(self.col_stayid))
              .pipe(self.pl_prepare_tstable, 
                    itemid_label='variableid',
                    col_intime='admissiontime',
                    col_measuretime='datetime',
                    id_mapping=self.id_mapping['observation'],
                    col_value='value',
                    )
              )
        # aaa=lf.collect()
        # # print(f'调试col_stay={self.col_stayid}')
        # # #=admissionid
        # print(f'调试lf={aaa}')
        self.save(lf, self.ts_savepth)
        return lf
    
    
    def gen_medication(self):
        self.get_labels(lazy=True)
        #self.labels = pl.scan_parquetE:\Science\Blended\hirid_data\labels.parquet)
        labels = self.labels.select('admissionid',
                                    'lengthofstay',
                                    'admissiontime')
        # self.med_parquet_path = E:/Science/Blended/hirid_data/raw_parquet/pharma_records.parquet
        # aaa=self.id_mapping
        # print(f'self.idmapping类型={type(aaa)}')
        # print(f'self.idmapping[pharma]类型={type(aaa['pharma'])}')
        # print(f'self.idmapping[pharma]={aaa['pharma']}')
        # for key,value in aaa['pharma'].items():
        #     print(f'key={type(key)},value={type(value)}')
        # #字典
        # print(f'pharma最终保存前的列:{zzz.columns}')
        # print(f'最终保存前的schema:{zzz.schema}')\
        str_key_mapping = {str(k): v for k, v in self.id_mapping['pharma'].items()}
        #最终保存前的schema:Schema({'patientid': Int32, 'pharmaid': Int32, 'givenat': Datetime(time_unit='ns', time_zone=None), 'givendose': Float32, 'doseunit': String, 'route': String})
        pharma = (pl.scan_parquet(self.med_parquet_path                )
                  .select('patientid',
                          'pharmaid',
                          'givenat',
                          'givendose',
                          'doseunit',
                          'route')
                  .with_columns(
                      pl.col('pharmaid')
                      .cast(pl.Utf8).replace(str_key_mapping)
                      # .map_elements(lambda x: self.id_mapping['pharma'].get(x, str(x)))
                      .alias('pharmaitem')
                      )
                  .drop('pharmaid')
                  .rename({'patientid': 'admissionid'}))
        # zzz=pharma.collect()
        # print(zzz)
   
        




        # med_par = (pl.scan_parquet(self.med_parquet_path).select('patientid',
        #                   'pharmaid',
        #                   'givenat',
        #                   'givendose',
        #                   'doseunit',
        #                   'route'))
        # aaa=med_par.collect()
        # print(f'调试={aaa},pharmid类型={aaa['pharmaid'].dtype},scheme={med_par.schema}')
        #pharmid类型=Int32,
        # print(f'self.id_mapping[pharm]={self.id_mapping['pharma']}')
        #scheme=Schema({'patientid': Int32, 'pharmaid': Int32, 'givenat': Datetime(time_unit='ns', time_zone=None), 'givendose': Float32, 'doseunit': String, 'route': String})
        dose_unit_conversions = {
            'g': {"omop_code": "mg",
                      "mul": 1e3},
            'µg': {'omop_code': 'mg',
                    'mul': 0.001},
            }

        # print(f'self.variable_ref_path ={self.variable_ref_path},self.admissions_path ={self.admissions_path},self.ts_path = {self.ts_path},self.med_path = {self.med_path}')
        # print(f'调试,pharma={pharma.collect()}')
        self.nmp = NewMedicationProcessor('hirid',
                                          lf_med=pharma,
                                          lf_labels=labels,
                                          col_pid='admissionid',
                                          col_med='pharmaitem',
                                          col_start='givenat',
                                          col_end=None,
                                          col_los='lengthofstay',
                                          col_dose='givendose',
                                          col_dose_unit='doseunit',
                                          col_route='route',
                                          offset_calc=True,
                                          col_admittime='admissiontime',
                                          dose_unit_conversion_dic=dose_unit_conversions
                                        )

        self.med = self.nmp.run()

        # print(f'大调试，self.med={self.med.explain(optimized=True)}')
        self.save(self.med, self.med_savepath)
