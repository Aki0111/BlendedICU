from pathlib import Path

import pandas as pd

class OMOP_Medications:
    def __init__(self, pth_dic, ingredients=None):
        '''A list of ingredients may be given as an input. 
        If ingredients is None, the 
        /medication_ingredients.csv
        file will be used as the list of ingredients.'''
        #用于将源数据库的药品名称映射到OMOP的vocabulary中
        self.pth_to_voc = pth_dic['vocabulary']
        self.pth_user_input = pth_dic['user_input']
        self.savedir = pth_dic['medication_mapping_files']
        self.savepath = self.savedir+'ohdsi_icu_medications.csv'
        
        self.concept = self._load_concept()
        #加载concept的concept_id和concept_name
        self.relations = self._load_concept_relationship()
        #加载concept_relationship加载concept_relationship的concept_id_1, concept_id_2, relationship_id
        self.ingredients = self._get_ingredients(ingredients)
        #药物成分列表，
        #如果 ingredients 为 None，则从 user_input/medication_ingredients.csv 读取 ingredient 列，转为列表。	#否则直接使用传入的 ingredients。
        self._check_unique_ingredients()
        #检查药物唯一性
    def _load_concept(self):
        print('Loading CONCEPT table...')
        concept_pth = self.pth_to_voc+'CONCEPT.parquet'
        concept = pd.read_parquet(concept_pth,
                                  columns=['concept_id', 'concept_name'])
        return concept
    
    def _load_concept_relationship(self):
        print('Loading CONCEPT_RELATIONSHIP table...')
        relationship_pth = self.pth_to_voc+'CONCEPT_RELATIONSHIP.parquet'
        concept_relationship = pd.read_parquet(relationship_pth,
                                               columns=['concept_id_1',
                                                        'concept_id_2',
                                                        'relationship_id'])
        return concept_relationship
    
    
    def _get_ingredients(self, ingredients):
        """
        If ingredient is None: read ingredient file.
        else: use the ingredient list provided"""
        if ingredients is None: 
            ingredients_pth = self.pth_user_input+'medication_ingredients.csv'    
            print(f'Loading ingredients from {ingredients_pth}')
            df = pd.read_csv(ingredients_pth)
            ingredients = df.ingredient.to_list()
        return ingredients
        
        
    def _check_unique_ingredients(self):
        vcounts = pd.Series(self.ingredients).value_counts()
        if vcounts.max() > 1:
            raise ValueError(f'Duplicate names in ingredients : '
                             f'{vcounts.loc[vcounts>1].to_list()}')

    def _ensure_vocabulary_is_downloaded(self):
        if not Path(self.pth_to_voc).is_dir():
            raise ValueError('Please download the RxNorm and RxNorm Extended'
                             'vocabularies from '
                             'https://athena.ohdsi.org/vocabulary/list')

    def run(self):
        med_idx = self.concept.concept_name.isin(self.ingredients)
        #isin（）方法返回一个布尔值的数组，表示每个元素是否在给定的列表中
        #med_idx为布尔索引，True表示该行满足条件，False表示该行不满足条件
        #这里检查concept_name 是否在 ingredients 列表中
        med_concept_ids = (self.concept.loc[med_idx]
                               .reset_index()
                               .set_index('concept_name'))
        #.loc[med_idx][] 根据med_idx布尔索引选择行
        #.reset_index() 重置索引
        #.set_index('concept_name') 将 concept_name 列设置为新的索引
        #效果为滤出 concept_name 在 ingredients 列表中的行，并将 concept_name 设为索引
        #之前的索引为 concept_id
        med_concept_ids.to_parquet(self.savedir+'med_concept_ids.parquet')
        print('Generating OMOP medication file...')
        keep_idx = self.relations.relationship_id == 'Has brand name'
        df = (self.relations.loc[keep_idx]
                            .drop(columns='relationship_id')
                            .reset_index(drop=True)
                            .rename(columns={'concept_id_1': 'ingredient_id',
                                             'concept_id_2': 'drug_id'}))
        #过滤 relations 表中 relationship_id 为 'Has brand name' 的行，重命名列为 ingredient_id 和 drug_id。
        
        df['ingredient'] = (self.concept.loc[df.ingredient_id]
                                        .reset_index(drop=True))
        df['drugname'] = (self.concept.loc[df.drug_id]
                                      .reset_index(drop=True))
        #用 ingredient_id 和 drug_id 去 concept 表查找成分名和药品名，分别赋值给 ingredient 和 drugname。
        df = df.loc[df.ingredient.isin(self.ingredients)]
        #只保留成分在成分列表的行。
        df = (df.pivot(columns='ingredient', values='drugname')
                .apply(lambda x: pd.Series(x.dropna().values)))
        #以成分为列，药品名为值，做 pivot 操作，得到成分到药品名的映射表。
        print(f'Saving {self.savepath}')
        df.to_csv(self.savepath, sep=';', index=None)
        self.ingredient_to_drug = df
        return self.ingredient_to_drug
        #得到成分到药品名的映射表。保存为 CSV 文件，并赋值给 self.ingredient_to_drug，最后返回。
        #表名为 ohdsi_icu_medications.csv
        