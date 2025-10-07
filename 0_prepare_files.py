'''
This code generates the mapping between ingredients from the OMOP standard
vocabulary and the drugnames in the source databases.

The omop_medication module creates a ohdsi_icu_medications.csv table
which contains brand names for a number of ingredients.
This file can be completed by a manual_icu_meds.csv file that lists additional
ingredients or additional synonyms or brand names for an ingredient in the
ohdsi vocabulary.

The medication_mapping module searches the labels in the source databases
and creates a json file listing all labels associated to an ingredient for each
source database.
'''
import json

from omopize.omop_medications import OMOP_Medications
from omopize.medication_mapping import MedicationMapping
from omopize.omop_diagnoses import DiagnosesMapping
from omop_cdm import omop_parquet 

pth_dic = json.load(open('paths.json', 'r'))

omop_parquet.convert_to_parquet(pth_dic)
#转化为parquet，Concept 和Concept_relationship，索引为concept_id
om = OMOP_Medications(pth_dic)

ingredient_to_drug = om.run()
#生成成分到药品名的映射表，并保存为 CSV 文件。
#表面名为 ohdsi_icu_medications.csv
mm = MedicationMapping(pth_dic,
                       datasets=[
                           'hirid', 
                                 #'amsterdam',
                                 'mimic4',
                       #          'mimic3',
                                 'eicu'])
#解压

medication_json = mm.run(load_drugnames=False, fname='medications.json')
        #{
#   "标准药物名1": {
#     "blended": 123456,
#     "amsterdam": ["别名A", "别名B", ...],
#     "eicu": ["别名C", ...],
#     "hirid": [...],
#     "mimic3": [...],
#     "mimic4": [...]
#   },
#   "标准药物名2": {
#     ...
#   }
# }
# blended 字段的值是一个整数，来源于 med_concept_ids.parquet 文件中该药物名的 concept_id。
# 如果找不到 concept_id，则为 0。
# 这个 concept_id 是 OMOP/OHDSI 标准词汇体系中该药物的唯一标识符。
dm = DiagnosesMapping(pth_dic, datasets=['mimic4'])
