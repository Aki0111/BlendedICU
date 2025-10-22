"""
This code extracts the data from the Amsterdam dataset 
('eicu_source_path' in paths.json).

It creates a set of .parquet files at the specified path 
('eicu' in paths.json). 
Approximate running time: 
    * raw_tables_to_parquet()  12min #only run once. csv.gz -> parquet with no data changes.
    * gen_* : 7min
"""
from eicu_preprocessing.eicupreparator import eicuPreparator

eicu_prep = eicuPreparator(
    physicalexam_pth='physicalExam.csv.gz',
    diag_pth='diagnosis.csv.gz',
    pasthistory_pth='pastHistory.csv.gz',
    admissiondx_pth='admissionDx.csv.gz',
    medication_pth='medication.csv.gz',
    infusiondrug_pth='infusionDrug.csv.gz',
    admissiondrug_pth='admissionDrug.csv.gz',
    patient_pth='patient.csv.gz',
    lab_pth='lab.csv.gz',
    respiratorycharting_pth='respiratoryCharting.csv.gz',
    nursecharting_pth='nurseCharting.csv.gz',
    periodic_pth='vitalPeriodic.csv.gz',
    aperiodic_pth='vitalAperiodic.csv.gz',
    intakeoutput_pth='intakeOutput.csv.gz')
print('raw_tables_to_parquet')
eicu_prep.raw_tables_to_parquet()
print('init_gen')
eicu_prep.init_gen()
print('gen_labels')
eicu_prep.gen_labels()
print('gen_flat')
eicu_prep.gen_flat()
print('gen_medication')
eicu_prep.gen_medication()
print('gen_timeseries')
eicu_prep.gen_timeseriesintakeoutput()
eicu_prep.gen_timeseriesresp()
eicu_prep.gen_timeserieslab()
eicu_prep.gen_timeseriesnurse()
eicu_prep.gen_timeseriesaperiodic()
eicu_prep.gen_timeseriesperiodic()
