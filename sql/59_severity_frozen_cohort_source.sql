
-- Outcome-blind MIMIC design cohort. All covariates end at or before t0.
CREATE TEMP TABLE _drug_map(itemid bigint, drug text);
INSERT INTO _drug_map VALUES
 (222168,'propofol'),(225150,'dexmedetomidine'),(229420,'dexmedetomidine'),
 (221668,'midazolam'),(221744,'fentanyl'),(225942,'fentanyl'),
 (221833,'hydromorphone'),(225154,'morphine'),
 (221555,'cisatracurium'),(229233,'rocuronium'),(222062,'vecuronium'),
 (221906,'norepinephrine'),(221749,'phenylephrine'),(229630,'phenylephrine'),
 (229632,'phenylephrine'),(222315,'vasopressin'),(221289,'epinephrine'),
 (229617,'epinephrine'),(221662,'dopamine');

CREATE TEMP TABLE _events AS
SELECT i.stay_id,i.subject_id,i.hadm_id,m.drug,i.starttime,i.endtime,
       i.rate,i.rateuom,i.statusdescription,i.orderid,i.linkorderid
FROM mimiciv_icu.inputevents i
JOIN _drug_map m USING(itemid)
WHERE i.rate>0 AND i.endtime>i.starttime
  AND COALESCE(i.statusdescription,'')<>'Bolus';
CREATE INDEX ON _events(stay_id,drug,starttime,endtime);

CREATE TEMP TABLE _target_times AS
SELECT stay_id,
       MIN(starttime) FILTER (WHERE drug='dexmedetomidine') AS dex_t,
       MIN(starttime) FILTER (WHERE drug='midazolam') AS mid_t
FROM _events GROUP BY stay_id;

CREATE TEMP TABLE _candidate AS
SELECT d.subject_id,d.hadm_id,d.stay_id,d.gender,d.admission_age,d.race,
       d.icu_intime,d.icu_outtime,d.first_icu_stay,d.hospital_expire_flag,
       d.los_icu,d.los_hospital,d.dod,a.admission_type,p.anchor_year_group,
       s.first_careunit,s.last_careunit,
       LEAST(t.dex_t,t.mid_t) AS t0,
       CASE WHEN t.dex_t=t.mid_t THEN 'TIE'
            WHEN t.dex_t IS NOT NULL AND (t.mid_t IS NULL OR t.dex_t<t.mid_t) THEN 'DEX'
            WHEN t.mid_t IS NOT NULL AND (t.dex_t IS NULL OR t.mid_t<t.dex_t) THEN 'MIDAZ' END AS treatment
FROM _target_times t
JOIN mimiciv_derived.icustay_detail d ON d.stay_id=t.stay_id
JOIN mimiciv_hosp.admissions a ON a.hadm_id=d.hadm_id
JOIN mimiciv_hosp.patients p ON p.subject_id=d.subject_id
JOIN mimiciv_icu.icustays s ON s.stay_id=d.stay_id
WHERE d.admission_age>=18
  AND LEAST(t.dex_t,t.mid_t)>=d.icu_intime
  AND LEAST(t.dex_t,t.mid_t)<=d.icu_outtime;
CREATE INDEX ON _candidate(stay_id);
CREATE INDEX ON _candidate(hadm_id);

CREATE TEMP TABLE _prop AS
SELECT c.stay_id,
       BOOL_OR(e.starttime<=c.t0 AND e.endtime>c.t0) AS prop_active_t0,
       COUNT(*) FILTER (WHERE e.endtime>c.t0-interval '2 hour' AND e.starttime<c.t0) AS prop_n_2h,
       COUNT(*) FILTER (WHERE e.endtime>c.t0-interval '4 hour' AND e.starttime<c.t0) AS prop_n_4h,
       COUNT(*) FILTER (WHERE e.endtime>c.t0-interval '6 hour' AND e.starttime<c.t0) AS prop_n_6h,
       MAX(e.starttime) FILTER (WHERE e.starttime<c.t0) AS last_prop_start,
       MAX(e.endtime) FILTER (WHERE e.starttime<c.t0) AS last_prop_end,
       MIN(e.starttime) FILTER (WHERE e.starttime<c.t0) AS first_prop_start,
       MAX(e.rate) FILTER (WHERE e.starttime<=c.t0 AND e.endtime>c.t0 AND e.rateuom='mcg/kg/min') AS prop_rate_t0,
       SUM(EXTRACT(epoch FROM (LEAST(e.endtime,c.t0)-GREATEST(e.starttime,c.t0-interval '24 hour')))/3600.0)
         FILTER (WHERE e.endtime>c.t0-interval '24 hour' AND e.starttime<c.t0) AS prop_overlap_hours_24h,
       SUM(e.rate*EXTRACT(epoch FROM (LEAST(e.endtime,c.t0)-GREATEST(e.starttime,c.t0-interval '2 hour'))))
         FILTER (WHERE e.rateuom='mcg/kg/min' AND e.endtime>c.t0-interval '2 hour' AND e.starttime<c.t0)
       / NULLIF(SUM(EXTRACT(epoch FROM (LEAST(e.endtime,c.t0)-GREATEST(e.starttime,c.t0-interval '2 hour'))))
         FILTER (WHERE e.rateuom='mcg/kg/min' AND e.endtime>c.t0-interval '2 hour' AND e.starttime<c.t0),0) AS prop_rate_mean_0_2h,
       SUM(e.rate*EXTRACT(epoch FROM (LEAST(e.endtime,c.t0-interval '2 hour')-GREATEST(e.starttime,c.t0-interval '6 hour'))))
         FILTER (WHERE e.rateuom='mcg/kg/min' AND e.endtime>c.t0-interval '6 hour' AND e.starttime<c.t0-interval '2 hour')
       / NULLIF(SUM(EXTRACT(epoch FROM (LEAST(e.endtime,c.t0-interval '2 hour')-GREATEST(e.starttime,c.t0-interval '6 hour'))))
         FILTER (WHERE e.rateuom='mcg/kg/min' AND e.endtime>c.t0-interval '6 hour' AND e.starttime<c.t0-interval '2 hour'),0) AS prop_rate_mean_2_6h
FROM _candidate c LEFT JOIN _events e ON e.stay_id=c.stay_id AND e.drug='propofol'
 AND e.starttime<c.t0 AND e.endtime>c.t0-interval '24 hour'
GROUP BY c.stay_id;

CREATE TEMP TABLE _drug_context AS
SELECT c.stay_id,
 BOOL_OR(e.drug IN ('fentanyl','hydromorphone','morphine') AND e.starttime<=c.t0 AND e.endtime>c.t0) AS opioid_active_t0,
 BOOL_OR(e.drug='fentanyl' AND e.starttime<=c.t0 AND e.endtime>c.t0) AS fentanyl_active_t0,
 BOOL_OR(e.drug='hydromorphone' AND e.starttime<=c.t0 AND e.endtime>c.t0) AS hydromorphone_active_t0,
 BOOL_OR(e.drug='morphine' AND e.starttime<=c.t0 AND e.endtime>c.t0) AS morphine_active_t0,
 COUNT(DISTINCT e.drug) FILTER (WHERE e.drug IN ('fentanyl','hydromorphone','morphine') AND e.endtime>c.t0-interval '6 hour' AND e.starttime<c.t0) AS opioid_types_6h,
 COUNT(*) FILTER (WHERE e.drug IN ('fentanyl','hydromorphone','morphine') AND e.starttime>=c.t0-interval '6 hour' AND e.starttime<c.t0) AS opioid_starts_6h,
 MAX(e.rate) FILTER (WHERE e.drug='fentanyl' AND e.starttime<=c.t0 AND e.endtime>c.t0 AND e.rateuom='mcg/hour') AS fentanyl_rate_mcg_h_t0,
 BOOL_OR(e.drug IN ('cisatracurium','rocuronium','vecuronium') AND e.starttime<=c.t0 AND e.endtime>c.t0) AS nmba_active_t0,
 BOOL_OR(e.drug IN ('cisatracurium','rocuronium','vecuronium') AND e.endtime>c.t0-interval '2 hour' AND e.starttime<c.t0) AS nmba_recent_2h,
 BOOL_OR(e.drug IN ('cisatracurium','rocuronium','vecuronium') AND e.endtime>c.t0-interval '4 hour' AND e.starttime<c.t0) AS nmba_recent_4h,
 BOOL_OR(e.drug IN ('cisatracurium','rocuronium','vecuronium') AND e.endtime>c.t0-interval '6 hour' AND e.starttime<c.t0) AS nmba_recent_6h,
 BOOL_OR(e.drug IN ('norepinephrine','phenylephrine','vasopressin','epinephrine','dopamine') AND e.starttime<=c.t0 AND e.endtime>c.t0) AS vasopressor_active_t0,
 COUNT(DISTINCT e.drug) FILTER (WHERE e.drug IN ('norepinephrine','phenylephrine','vasopressin','epinephrine','dopamine') AND e.starttime<=c.t0 AND e.endtime>c.t0) AS vasopressor_count_t0
FROM _candidate c LEFT JOIN _events e ON e.stay_id=c.stay_id
 AND e.starttime<c.t0 AND e.endtime>c.t0-interval '6 hour'
GROUP BY c.stay_id;

CREATE TEMP TABLE _sed AS
SELECT c.stay_id,
 (ARRAY_AGG(CASE WHEN ce.itemid=228096 THEN 'RASS' ELSE 'SAS' END ORDER BY ce.charttime DESC))[1] AS sedation_scale,
 (ARRAY_AGG(ce.valuenum ORDER BY ce.charttime DESC))[1] AS sedation_score,
 MAX(ce.charttime) AS sedation_time,
 COUNT(*) AS sedation_n_6h
FROM _candidate c JOIN mimiciv_icu.chartevents ce ON ce.stay_id=c.stay_id
 AND ce.charttime<=c.t0 AND ce.charttime>=c.t0-interval '6 hour'
 AND ((ce.itemid=228096 AND ce.valuenum BETWEEN -5 AND 4)
   OR (ce.itemid=223753 AND ce.valuenum BETWEEN 1 AND 7))
GROUP BY c.stay_id;

CREATE TEMP TABLE _gcs AS
SELECT c.stay_id,(ARRAY_AGG(g.gcs ORDER BY g.charttime DESC))[1] AS gcs_6h
FROM _candidate c JOIN mimiciv_derived.gcs g ON g.stay_id=c.stay_id
 AND g.charttime<=c.t0 AND g.charttime>=c.t0-interval '6 hour'
GROUP BY c.stay_id;

CREATE TEMP TABLE _mv AS
SELECT c.stay_id,
 BOOL_OR(v.ventilation_status='InvasiveVent' AND c.t0>=v.starttime AND c.t0<v.endtime) AS invasive_mv_t0,
 MIN(v.starttime) FILTER (WHERE v.ventilation_status='InvasiveVent' AND c.t0>=v.starttime AND c.t0<v.endtime) AS current_mv_start,
 MIN(v.endtime) FILTER (WHERE v.ventilation_status='InvasiveVent' AND c.t0>=v.starttime AND c.t0<v.endtime) AS current_mv_end
FROM _candidate c LEFT JOIN mimiciv_derived.ventilation v ON v.stay_id=c.stay_id
GROUP BY c.stay_id;

CREATE TEMP TABLE _vent AS
SELECT c.stay_id,
 (ARRAY_AGG(v.ventilator_mode ORDER BY v.charttime DESC) FILTER (WHERE v.ventilator_mode IS NOT NULL))[1] AS ventilator_mode,
 (ARRAY_AGG(v.ventilator_mode_hamilton ORDER BY v.charttime DESC) FILTER (WHERE v.ventilator_mode_hamilton IS NOT NULL))[1] AS ventilator_mode_hamilton,
 (ARRAY_AGG(v.peep ORDER BY v.charttime DESC) FILTER (WHERE v.peep IS NOT NULL))[1] AS peep,
 (ARRAY_AGG(v.fio2 ORDER BY v.charttime DESC) FILTER (WHERE v.fio2 IS NOT NULL))[1] AS fio2,
 (ARRAY_AGG(v.respiratory_rate_spontaneous ORDER BY v.charttime DESC) FILTER (WHERE v.respiratory_rate_spontaneous IS NOT NULL))[1] AS rr_spont,
 (ARRAY_AGG(v.respiratory_rate_set ORDER BY v.charttime DESC) FILTER (WHERE v.respiratory_rate_set IS NOT NULL))[1] AS rr_set
FROM _candidate c JOIN mimiciv_derived.ventilator_setting v ON v.stay_id=c.stay_id
 AND v.charttime<=c.t0 AND v.charttime>=c.t0-interval '4 hour'
GROUP BY c.stay_id;

CREATE TEMP TABLE _vitals AS
SELECT c.stay_id,
 (ARRAY_AGG(v.heart_rate ORDER BY v.charttime DESC) FILTER (WHERE v.heart_rate IS NOT NULL))[1] AS heart_rate,
 (ARRAY_AGG(COALESCE(v.mbp,v.mbp_ni) ORDER BY v.charttime DESC) FILTER (WHERE COALESCE(v.mbp,v.mbp_ni) IS NOT NULL))[1] AS map,
 (ARRAY_AGG(COALESCE(v.sbp,v.sbp_ni) ORDER BY v.charttime DESC) FILTER (WHERE COALESCE(v.sbp,v.sbp_ni) IS NOT NULL))[1] AS sbp,
 MIN(v.heart_rate) AS heart_rate_min_6h, MIN(COALESCE(v.mbp,v.mbp_ni)) AS map_min_6h,
 MIN(COALESCE(v.sbp,v.sbp_ni)) AS sbp_min_6h
FROM _candidate c JOIN mimiciv_derived.vitalsign v ON v.stay_id=c.stay_id
 AND v.charttime<=c.t0 AND v.charttime>=c.t0-interval '6 hour'
GROUP BY c.stay_id;

CREATE TEMP TABLE _chem AS
SELECT c.stay_id,
 (ARRAY_AGG(x.creatinine ORDER BY x.charttime DESC) FILTER (WHERE x.creatinine IS NOT NULL))[1] AS creatinine,
 (ARRAY_AGG(x.bun ORDER BY x.charttime DESC) FILTER (WHERE x.bun IS NOT NULL))[1] AS bun,
 (ARRAY_AGG(x.bicarbonate ORDER BY x.charttime DESC) FILTER (WHERE x.bicarbonate IS NOT NULL))[1] AS bicarbonate
FROM _candidate c JOIN mimiciv_derived.chemistry x ON x.hadm_id=c.hadm_id
 AND x.charttime<=c.t0 AND x.charttime>=c.t0-interval '24 hour'
GROUP BY c.stay_id;

CREATE TEMP TABLE _enz AS
SELECT c.stay_id,
 (ARRAY_AGG(x.bilirubin_total ORDER BY x.charttime DESC) FILTER (WHERE x.bilirubin_total IS NOT NULL))[1] AS bilirubin
FROM _candidate c JOIN mimiciv_derived.enzyme x ON x.hadm_id=c.hadm_id
 AND x.charttime<=c.t0 AND x.charttime>=c.t0-interval '24 hour'
GROUP BY c.stay_id;

CREATE TEMP TABLE _bg AS
SELECT c.stay_id,
 (ARRAY_AGG(x.lactate ORDER BY x.charttime DESC) FILTER (WHERE x.lactate IS NOT NULL))[1] AS lactate,
 (ARRAY_AGG(x.pao2fio2ratio ORDER BY x.charttime DESC) FILTER (WHERE x.pao2fio2ratio IS NOT NULL))[1] AS pf_ratio
FROM _candidate c JOIN mimiciv_derived.bg x ON x.hadm_id=c.hadm_id
 AND x.charttime<=c.t0 AND x.charttime>=c.t0-interval '12 hour'
GROUP BY c.stay_id;

SELECT c.subject_id::text AS patient_id,c.hadm_id::text AS admission_id,c.stay_id::text AS stay_id,
 c.treatment,c.t0::text AS time_zero,
 EXTRACT(epoch FROM (c.t0-c.icu_intime))/3600.0 AS hours_from_icu_admit,
 c.admission_age::double precision AS age,c.gender AS sex,c.race,c.first_icu_stay::int,
 c.admission_type,c.anchor_year_group,c.first_careunit,c.last_careunit,
 COALESCE(p.prop_active_t0,false)::int AS prop_active_t0,p.prop_n_2h,p.prop_n_4h,p.prop_n_6h,
 CASE WHEN p.last_prop_start IS NOT NULL THEN EXTRACT(epoch FROM (c.t0-p.last_prop_start))/3600.0 END AS hours_since_last_prop_start,
 CASE WHEN p.last_prop_end IS NOT NULL THEN EXTRACT(epoch FROM (c.t0-p.last_prop_end))/3600.0 END AS hours_since_last_prop_end,
 CASE WHEN p.first_prop_start IS NOT NULL THEN EXTRACT(epoch FROM (c.t0-p.first_prop_start))/3600.0 END AS hours_since_first_prop_start,
 p.prop_rate_t0,p.prop_overlap_hours_24h,p.prop_rate_mean_0_2h,p.prop_rate_mean_2_6h,
 COALESCE(dc.opioid_active_t0,false)::int AS opioid_active_t0,
 COALESCE(dc.fentanyl_active_t0,false)::int AS fentanyl_active_t0,
 COALESCE(dc.hydromorphone_active_t0,false)::int AS hydromorphone_active_t0,
 COALESCE(dc.morphine_active_t0,false)::int AS morphine_active_t0,
 dc.opioid_types_6h,dc.opioid_starts_6h,dc.fentanyl_rate_mcg_h_t0,
 COALESCE(dc.nmba_active_t0,false)::int AS nmba_active_t0,
 COALESCE(dc.nmba_recent_2h,false)::int AS nmba_recent_2h,
 COALESCE(dc.nmba_recent_4h,false)::int AS nmba_recent_4h,
 COALESCE(dc.nmba_recent_6h,false)::int AS nmba_recent_6h,
 COALESCE(dc.vasopressor_active_t0,false)::int AS vasopressor_active_t0,dc.vasopressor_count_t0,
 s.sedation_scale,s.sedation_score,s.sedation_n_6h,
 CASE WHEN s.sedation_time IS NOT NULL THEN EXTRACT(epoch FROM (c.t0-s.sedation_time))/3600.0 END AS hours_since_sedation,
 g.gcs_6h,COALESCE(m.invasive_mv_t0,false)::int AS invasive_mv_t0,
 CASE WHEN m.current_mv_start IS NOT NULL THEN EXTRACT(epoch FROM (c.t0-m.current_mv_start))/3600.0 END AS hours_since_mv_start,
 CASE WHEN m.current_mv_end IS NOT NULL THEN EXTRACT(epoch FROM (m.current_mv_end-c.t0))/3600.0 END AS hours_to_current_mv_end,
 v.ventilator_mode,v.ventilator_mode_hamilton,v.peep,v.fio2,v.rr_spont,v.rr_set,
 vs.heart_rate,vs.map,vs.sbp,vs.heart_rate_min_6h,vs.map_min_6h,vs.sbp_min_6h,
 ch.creatinine,ch.bun,ch.bicarbonate,en.bilirubin,bg.lactate,bg.pf_ratio,
 c.hospital_expire_flag AS hospital_mortality,c.los_icu,c.los_hospital,c.dod::text AS dod,
 c.icu_outtime::text AS icu_outtime
FROM _candidate c
LEFT JOIN _prop p USING(stay_id) LEFT JOIN _drug_context dc USING(stay_id)
LEFT JOIN _sed s USING(stay_id) LEFT JOIN _gcs g USING(stay_id)
LEFT JOIN _mv m USING(stay_id) LEFT JOIN _vent v USING(stay_id)
LEFT JOIN _vitals vs USING(stay_id) LEFT JOIN _chem ch USING(stay_id)
LEFT JOIN _enz en USING(stay_id) LEFT JOIN _bg bg USING(stay_id)

ORDER BY c.stay_id;
