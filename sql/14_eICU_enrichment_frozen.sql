CREATE TEMP TABLE _cohort(stay_id bigint,t0 integer);


CREATE INDEX ON _cohort(stay_id);


CREATE TEMP TABLE _inf AS

SELECT c.stay_id,

 COUNT(*) FILTER(WHERE lower(i.drugname) LIKE '%propofol%' AND i.infusionoffset BETWEEN c.t0-15 AND c.t0) prop_n_15,

 COUNT(*) FILTER(WHERE lower(i.drugname) LIKE '%propofol%' AND i.infusionoffset BETWEEN c.t0-60 AND c.t0) prop_n_60,

 COUNT(*) FILTER(WHERE lower(i.drugname) LIKE '%propofol%' AND i.infusionoffset BETWEEN c.t0-360 AND c.t0) prop_n_360,

 MAX(i.infusionoffset) FILTER(WHERE lower(i.drugname) LIKE '%propofol%' AND i.infusionoffset<=c.t0) last_prop_t,

 BOOL_OR(lower(i.drugname) ~ '(fentanyl|hydromorphone|dilaudid|morphine)' AND i.infusionoffset BETWEEN c.t0-360 AND c.t0) opioid_6h,

 BOOL_OR(lower(i.drugname) ~ '(cisatracurium|nimbex|rocuronium|vecuronium)' AND i.infusionoffset BETWEEN c.t0-360 AND c.t0) nmba_6h,

 BOOL_OR(lower(i.drugname) ~ '(norepinephrine|levophed|phenylephrine|neosynephrine|neo-synephrine|vasopressin|epinephrine|dopamine)'

   AND i.infusionoffset BETWEEN c.t0-360 AND c.t0) vasopressor_pre6h,

 BOOL_OR(lower(i.drugname) ~ '(norepinephrine|levophed|phenylephrine|neosynephrine|neo-synephrine|vasopressin|epinephrine|dopamine)'

   AND i.infusionoffset > c.t0 AND i.infusionoffset <= c.t0+360) vasopressor_post6h

FROM _cohort c LEFT JOIN infusiondrug i ON i.patientunitstayid=c.stay_id

 AND i.infusionoffset BETWEEN c.t0-360 AND c.t0+360

 AND TRIM(i.drugrate) ~ '^-?[0-9]+(\.[0-9]+)?$' AND TRIM(i.drugrate)::numeric>0

GROUP BY c.stay_id;


CREATE TEMP TABLE _vital AS

SELECT c.stay_id,

 (ARRAY_AGG(v.heartrate ORDER BY v.observationoffset DESC) FILTER(WHERE v.observationoffset BETWEEN c.t0-360 AND c.t0 AND v.heartrate BETWEEN 20 AND 250))[1] hr_last,

 MIN(v.heartrate) FILTER(WHERE v.observationoffset BETWEEN c.t0-360 AND c.t0 AND v.heartrate BETWEEN 20 AND 250) hr_min_pre6h,

 (ARRAY_AGG(v.systemicmean ORDER BY v.observationoffset DESC) FILTER(WHERE v.observationoffset BETWEEN c.t0-360 AND c.t0 AND v.systemicmean BETWEEN 20 AND 180))[1] map_last,

 MIN(v.systemicmean) FILTER(WHERE v.observationoffset BETWEEN c.t0-360 AND c.t0 AND v.systemicmean BETWEEN 20 AND 180) map_min_pre6h,

 MIN(v.heartrate) FILTER(WHERE v.observationoffset>c.t0 AND v.observationoffset<=c.t0+360 AND v.heartrate BETWEEN 20 AND 250) hr_min_6h,

 MIN(v.heartrate) FILTER(WHERE v.observationoffset>c.t0 AND v.observationoffset<=c.t0+720 AND v.heartrate BETWEEN 20 AND 250) hr_min_12h,

 MIN(v.heartrate) FILTER(WHERE v.observationoffset>c.t0 AND v.observationoffset<=c.t0+1440 AND v.heartrate BETWEEN 20 AND 250) hr_min_24h,

 MIN(v.systemicmean) FILTER(WHERE v.observationoffset>c.t0 AND v.observationoffset<=c.t0+360 AND v.systemicmean BETWEEN 20 AND 180) map_min_6h,

 MIN(v.systemicmean) FILTER(WHERE v.observationoffset>c.t0 AND v.observationoffset<=c.t0+720 AND v.systemicmean BETWEEN 20 AND 180) map_min_12h,

 MIN(v.systemicmean) FILTER(WHERE v.observationoffset>c.t0 AND v.observationoffset<=c.t0+1440 AND v.systemicmean BETWEEN 20 AND 180) map_min_24h

FROM _cohort c LEFT JOIN vitalperiodic v ON v.patientunitstayid=c.stay_id

 AND v.observationoffset BETWEEN c.t0-360 AND c.t0+1440

GROUP BY c.stay_id;


CREATE TEMP TABLE _lab AS

SELECT c.stay_id,

 (ARRAY_AGG(l.labresult ORDER BY l.labresultoffset DESC) FILTER(WHERE lower(l.labname)='creatinine' AND l.labresult BETWEEN .1 AND 20))[1] creatinine,

 (ARRAY_AGG(l.labresult ORDER BY l.labresultoffset DESC) FILTER(WHERE lower(l.labname) LIKE '%bilirubin%total%' AND l.labresult BETWEEN 0 AND 60))[1] bilirubin,

 (ARRAY_AGG(l.labresult ORDER BY l.labresultoffset DESC) FILTER(WHERE lower(l.labname)='lactate' AND l.labresult BETWEEN 0 AND 30))[1] lactate

FROM _cohort c LEFT JOIN lab l ON l.patientunitstayid=c.stay_id

 AND l.labresultoffset BETWEEN c.t0-1440 AND c.t0

GROUP BY c.stay_id;


CREATE TEMP TABLE _dx AS

SELECT c.stay_id,

 BOOL_OR(lower(d.diagnosisstring) ~ '(seizure|status epilep|epilepsy)') seizure,

 BOOL_OR(lower(d.diagnosisstring) ~ '(alcohol.*withdraw|withdrawal.*alcohol|benzodiazepine.*withdraw)') withdrawal,

 BOOL_OR(lower(d.diagnosisstring) ~ '(intracranial|subarachnoid|stroke|traumatic brain|head injury|cerebral edema)') neurocritical,

 BOOL_OR(lower(d.diagnosisstring) ~ '(respiratory failure|acute respiratory distress|pneumonia|aspiration|pulmonary edema)') respiratory,

 BOOL_OR(lower(d.diagnosisstring) ~ '(delirium|agitation|encephalopathy)') delirium_proxy,

 BOOL_OR(lower(d.diagnosisstring) ~ '(postoperative|postprocedural|surgery)') postoperative

FROM _cohort c LEFT JOIN diagnosis d ON d.patientunitstayid=c.stay_id AND d.diagnosisoffset BETWEEN -1440 AND c.t0

GROUP BY c.stay_id;


CREATE TEMP TABLE _tx AS

SELECT c.stay_id,

 BOOL_OR(lower(t.treatmentstring) LIKE '%mechanical ventilation%' AND t.treatmentoffset BETWEEN c.t0-1440 AND c.t0) mv_treatment_24h

FROM _cohort c LEFT JOIN treatment t ON t.patientunitstayid=c.stay_id AND t.treatmentoffset BETWEEN c.t0-1440 AND c.t0

GROUP BY c.stay_id;


SELECT c.stay_id::text,p.hospitaldischargeyear,p.apacheadmissiondx,

 COALESCE(i.prop_n_15,0) prop_n_15,COALESCE(i.prop_n_60,0) prop_n_60,COALESCE(i.prop_n_360,0) prop_n_360,

 CASE WHEN i.last_prop_t IS NOT NULL THEN c.t0-i.last_prop_t END minutes_since_last_prop,

 COALESCE(i.opioid_6h,false)::int opioid_6h,COALESCE(i.nmba_6h,false)::int nmba_6h,

 COALESCE(i.vasopressor_pre6h,false)::int vasopressor_pre6h,

 COALESCE(i.vasopressor_post6h,false)::int vasopressor_post6h,

 v.hr_last,v.hr_min_pre6h,v.map_last,v.map_min_pre6h,v.hr_min_6h,v.hr_min_12h,v.hr_min_24h,

 v.map_min_6h,v.map_min_12h,v.map_min_24h,l.creatinine,l.bilirubin,l.lactate,

 COALESCE(d.seizure,false)::int seizure,COALESCE(d.withdrawal,false)::int withdrawal,

 COALESCE(d.neurocritical,false)::int neurocritical,COALESCE(d.respiratory,false)::int respiratory,

 COALESCE(d.delirium_proxy,false)::int delirium_proxy,COALESCE(d.postoperative,false)::int postoperative,

 COALESCE(t.mv_treatment_24h,false)::int mv_treatment_24h,

 CASE WHEN lower(p.unitdischargestatus)='expired' THEN 1 WHEN lower(p.unitdischargestatus)='alive' THEN 0 END icu_mortality

FROM _cohort c JOIN patient p ON p.patientunitstayid=c.stay_id

LEFT JOIN _inf i USING(stay_id) LEFT JOIN _vital v USING(stay_id) LEFT JOIN _lab l USING(stay_id)

LEFT JOIN _dx d USING(stay_id) LEFT JOIN _tx t USING(stay_id)

ORDER BY c.stay_id;

