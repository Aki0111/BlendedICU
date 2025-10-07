"""
This code extracts the data from the Amsterdam dataset 
('hirid_source_path' in paths.json).

It creates a set of .parquet files at the specified path 
('hirid' in paths.json). 

"""
import os
from hirid_preprocessing.HiridPreparator import hiridPreparator

hirid_prep = hiridPreparator(
    variable_ref_path='reference_data/hirid_variable_reference.csv',
    ts_path='observation_tables/parquet/',
    pharma_path='pharma_records/parquet/',
    admissions_path='reference_data/general_table.csv',
    imputedstage_path='imputed_stage/parquet/')

hirid_prep.raw_tables_to_parquet()

hirid_prep.init_gen()

if os.path.exists("hirid_data/labels.parquet"):
    print(f"文件hirid_data/labels.parquet存在,不再生成")
else:
    print(f"文件hirid_data/labels.parquet不存在,开始生成")
    hirid_prep.gen_labels()
if os.path.exists("hirid_data/medications.parquet"):
    print(f"文件hirid_data/medications.parquet存在,不再生成")
else:
    print(f"文件hirid_data/medications.parquet不存在,开始生成")
    hirid_prep.gen_medication()
if os.path.exists("hirid_data/timeseries.parquet"):
    print(f"文件hirid_data/timeseries.parquet存在,不再生成")
else:
    print(f"文件hirid_data/timeseries.parquet不存在,开始生成")
    hirid_prep.gen_timeseries()
