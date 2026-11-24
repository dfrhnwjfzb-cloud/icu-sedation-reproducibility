CREATE TEMP TABLE _cohort(stay_id bigint,hadm_id bigint,subject_id bigint,t0 timestamp);


CREATE INDEX ON _cohort(stay_id); CREATE INDEX ON _cohort(hadm_id);


CREATE TEMP TABLE _lib AS

SELECT c.stay_id,

 MIN(v.endtime) FILTER (WHERE v.ventilation_status='InvasiveVent' AND v.endtime>c.t0

   AND v.endtime<=c.t0+interval '28 day'

   AND NOT EXISTS (SELECT 1 FROM mimiciv_derived.ventilation v2

       WHERE v2.stay_id=c.stay_id AND v2.ventilation_status='InvasiveVent'

         AND v2.starttime>=v.endtime AND v2.starttime<v.endtime+interval '48 hour')) AS sustainable_liberation_time,

 MIN(v.endtime) FILTER (WHERE v.ventilation_status='InvasiveVent' AND v.starttime<=c.t0 AND v.endtime>c.t0) AS initial_mv_end

FROM _cohort c LEFT JOIN mimiciv_derived.ventilation v ON v.stay_id=c.stay_id

GROUP BY c.stay_id;


CREATE TEMP TABLE _safety AS

SELECT c.stay_id,

 MIN(v.heart_rate) FILTER (WHERE v.charttime>c.t0 AND v.charttime<=c.t0+interval '24 hour') AS min_hr_24h,

 MIN(COALESCE(v.mbp,v.mbp_ni)) FILTER (WHERE v.charttime>c.t0 AND v.charttime<=c.t0+interval '24 hour') AS min_map_24h

FROM _cohort c LEFT JOIN mimiciv_derived.vitalsign v ON v.stay_id=c.stay_id

 AND v.charttime>c.t0 AND v.charttime<=c.t0+interval '24 hour'

GROUP BY c.stay_id;


CREATE TEMP TABLE _vaso AS

SELECT c.stay_id,

 BOOL_OR(v.starttime>c.t0 AND v.starttime<=c.t0+interval '6 hour') AS any_vasopressor_start_6h

FROM _cohort c LEFT JOIN mimiciv_derived.vasoactive_agent v ON v.stay_id=c.stay_id

 AND v.starttime>c.t0 AND v.starttime<=c.t0+interval '6 hour'

GROUP BY c.stay_id;


SELECT c.stay_id::text,l.sustainable_liberation_time::text,

 CASE WHEN l.sustainable_liberation_time IS NOT NULL

       AND (p.dod IS NULL OR p.dod>(c.t0+interval '28 day')::date) THEN 1 ELSE 0 END AS alive_sustainably_liberated_day28,

 CASE WHEN p.dod IS NOT NULL AND p.dod<=(c.t0+interval '28 day')::date THEN 1 ELSE 0 END AS mortality_day28,

 a.hospital_expire_flag AS hospital_mortality,

 CASE WHEN a.deathtime IS NOT NULL AND a.deathtime<=i.outtime THEN 1 ELSE 0 END AS icu_mortality,

 CASE WHEN l.sustainable_liberation_time IS NOT NULL THEN EXTRACT(epoch FROM (l.sustainable_liberation_time-c.t0))/86400.0 END AS days_to_sustainable_liberation,

 EXTRACT(epoch FROM (i.outtime-c.t0))/86400.0 AS icu_los_after_t0,

 EXTRACT(epoch FROM (a.dischtime-c.t0))/86400.0 AS hospital_los_after_t0,

 CASE WHEN s.min_hr_24h<50 THEN 1 ELSE 0 END AS bradycardia_24h,

 CASE WHEN s.min_map_24h<60 THEN 1 ELSE 0 END AS hypotension_24h,

 COALESCE(v.any_vasopressor_start_6h,false)::int AS vasopressor_start_6h,

 CASE WHEN EXISTS (SELECT 1 FROM mimiciv_derived.ventilation z

    WHERE z.stay_id=c.stay_id AND z.ventilation_status='InvasiveVent'

      AND l.initial_mv_end IS NOT NULL AND z.starttime>=l.initial_mv_end

      AND z.starttime<l.initial_mv_end+interval '7 day') THEN 1 ELSE 0 END AS reintubation_7d

FROM _cohort c JOIN mimiciv_icu.icustays i ON i.stay_id=c.stay_id

JOIN mimiciv_hosp.admissions a ON a.hadm_id=c.hadm_id

JOIN mimiciv_hosp.patients p ON p.subject_id=c.subject_id

LEFT JOIN _lib l ON l.stay_id=c.stay_id LEFT JOIN _safety s ON s.stay_id=c.stay_id

LEFT JOIN _vaso v ON v.stay_id=c.stay_id

ORDER BY c.stay_id;

