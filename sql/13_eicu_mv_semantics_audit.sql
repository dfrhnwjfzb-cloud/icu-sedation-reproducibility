CREATE TEMP TABLE _map(raw_name text,drug text);

INSERT INTO _map VALUES ('Dexmedetomidine','dexmedetomidine'),

('Dexmedetomidine (mcg/hr)','dexmedetomidine'),

('Dexmedetomidine (mcg/kg/hr)','dexmedetomidine'),

('Dexmedetomidine (mcg/kg/hr) ()','dexmedetomidine'),

('Dexmedetomidine (mcg/kg/hr) (mcg/kg/hr)','dexmedetomidine'),

('Dexmedetomidine (mcg/kg/min)','dexmedetomidine'),

('Dexmedetomidine (mcg/min)','dexmedetomidine'),

('Dexmedetomidine (mg/hr)','dexmedetomidine'),

('Dexmedetomidine (mg/kg/hr)','dexmedetomidine'),

('Dexmedetomidine (mg/kg/min)','dexmedetomidine'),

('Dexmedetomidine (ml/hr)','dexmedetomidine'),

('Dexmedetomidine (units/min)','dexmedetomidine'),

('Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml ()','dexmedetomidine'),

('Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml (mcg/kg/hr)','dexmedetomidine'),

('Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml (mcg/min)','dexmedetomidine'),

('Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml (ml/hr)','dexmedetomidine'),

('Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml ()','dexmedetomidine'),

('Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/kg/hr)','dexmedetomidine'),

('Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/kg/min)','dexmedetomidine'),

('Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (mcg/min)','dexmedetomidine'),

('Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (ml/hr)','dexmedetomidine'),

('Dexmedetomidine(Precedex) 400 mcg Sodium Chloride 0.9% 100 ml (mcg/kg/hr)','dexmedetomidine'),

('Dexmetetomidine (Precedex) (mcg/kg/hr)','dexmedetomidine'),

('Midazolam','midazolam'),

('Midazolam ()','midazolam'),

('Midazolam (Versed) 100 mg Sodium Chloride 0.9% 100 ml (mg/hr)','midazolam'),

('Midazolam (Versed) 100 mg Sodium Chloride 0.9% 350 ml (mg/hr)','midazolam'),

('Midazolam (mcg/hr)','midazolam'),

('Midazolam (mcg/kg/hr)','midazolam'),

('Midazolam (mcg/kg/min)','midazolam'),

('Midazolam (mg/hr)','midazolam'),

('Midazolam (mg/kg/hr)','midazolam'),

('Midazolam (ml/hr)','midazolam'),

('NS w/ propofol (ml/hr)','propofol'),

('NSS w/ precedex/fent (ml/hr)','dexmedetomidine'),

('NSS w/ versed/fent (ml/hr)','midazolam'),

('PRECEDEX (mcg/kg/hr)','dexmedetomidine'),

('PROPOFOL (mcg/kg/min)','propofol'),

('Precedex','dexmedetomidine'),

('Precedex  (mcg/kg/hr)','dexmedetomidine'),

('Precedex (mcg/kg/hr)','dexmedetomidine'),

('Precedex (mcg/kg/min)','dexmedetomidine'),

('Precedex (mg/kg/hr)','dexmedetomidine'),

('Precedex (ml/hr)','dexmedetomidine'),

('Propofol','propofol'),

('Propofol ()','propofol'),

('Propofol (Diprivan) 1000 mg  100 ml  Premix (mcg/kg/min)','propofol'),

('Propofol (Diprivan) 1000 mg  130 ml  Premix (mcg/kg/min)','propofol'),

('Propofol (Diprivan) 1000 mg Sterile Water (SWFI) 100 ml  Premix (mcg/kg/min)','propofol'),

('Propofol (Unknown)','propofol'),

('Propofol (mcg/hr)','propofol'),

('Propofol (mcg/kg/hr)','propofol'),

('Propofol (mcg/kg/min)','propofol'),

('Propofol (mcg/min)','propofol'),

('Propofol (mg/hr)','propofol'),

('Propofol (mg/kg/hr)','propofol'),

('Propofol (mg/kg/min)','propofol'),

('Propofol (mg/min)','propofol'),

('Propofol (ml/hr)','propofol'),

('Propofol IV Emulsion 100 ml (ml/hr)','propofol'),

('Versed','midazolam'),

('Versed (mg/hr)','midazolam'),

('Versed (ml/hr)','midazolam'),

('diprivan (mcg/min)','propofol'),

('precedex','dexmedetomidine'),

('precedex  (mcg/kg/hr)','dexmedetomidine'),

('precedex (mcg/kg/hr)','dexmedetomidine'),

('precedex (ml/hr)','dexmedetomidine'),

('propofol (mcg/kg/min)','propofol'),

('propofol (mcg/min)','propofol'),

('propofol (ml/hr)','propofol'),

('versed','midazolam'),

('versed (mg/hr)','midazolam'),

('versed (ml/hr)','midazolam');

CREATE TEMP TABLE _events AS

SELECT i.patientunitstayid AS stay_id,p.hospitalid,m.drug,i.infusionoffset AS t

FROM infusiondrug i JOIN _map m ON m.raw_name=i.drugname

JOIN patient p USING(patientunitstayid)

WHERE TRIM(i.drugrate) ~ '^-?[0-9]+(\.[0-9]+)?$'

  AND TRIM(i.drugrate)::numeric>0;

CREATE INDEX _events_stay_drug_time_idx ON _events(stay_id,drug,t);


CREATE TEMP TABLE _candidate AS

WITH tt AS (

  SELECT stay_id,MIN(t) FILTER(WHERE drug='dexmedetomidine') dex_t,

         MIN(t) FILTER(WHERE drug='midazolam') mid_t

  FROM _events GROUP BY stay_id

)

SELECT p.patientunitstayid AS stay_id,p.hospitalid,

       LEAST(tt.dex_t,tt.mid_t) AS t0,

       CASE WHEN tt.dex_t=tt.mid_t THEN 'TIE'

            WHEN tt.dex_t IS NOT NULL AND (tt.mid_t IS NULL OR tt.dex_t<tt.mid_t) THEN 'DEX'

            ELSE 'MIDAZ' END treatment

FROM tt JOIN patient p ON p.patientunitstayid=tt.stay_id

WHERE LEAST(tt.dex_t,tt.mid_t) BETWEEN 0 AND p.unitdischargeoffset;

CREATE INDEX _candidate_stay_idx ON _candidate(stay_id);


CREATE TEMP TABLE _airway AS

SELECT c.stay_id,

  (ARRAY_AGG(NULLIF(r.airwaytype,'') ORDER BY r.respcarestatusoffset DESC)

   FILTER(WHERE r.respcarestatusoffset BETWEEN c.t0-1440 AND c.t0))[1] last_airway_24h,

  MAX(r.respcarestatusoffset) FILTER(WHERE r.respcarestatusoffset BETWEEN c.t0-1440 AND c.t0) last_airway_offset_24h,

  BOOL_OR(r.airwaytype IN ('Oral ETT','Nasal ETT','Tracheostomy','Double-Lumen Tube','Cricothyrotomy')

          AND r.respcarestatusoffset BETWEEN c.t0-360 AND c.t0) airway_6h,

  BOOL_OR(r.airwaytype IN ('Oral ETT','Nasal ETT','Tracheostomy','Double-Lumen Tube','Cricothyrotomy')

          AND r.respcarestatusoffset BETWEEN c.t0-720 AND c.t0) airway_12h,

  BOOL_OR(r.airwaytype IN ('Oral ETT','Nasal ETT','Tracheostomy','Double-Lumen Tube','Cricothyrotomy')

          AND r.respcarestatusoffset BETWEEN c.t0-1440 AND c.t0) airway_24h,

  BOOL_OR(c.t0 BETWEEN r.ventstartoffset AND r.ventendoffset

          AND COALESCE(r.airwaytype,'')<>'No Artificial Airway') strict_interval

FROM _candidate c LEFT JOIN respiratorycare r ON r.patientunitstayid=c.stay_id

GROUP BY c.stay_id;


CREATE TEMP TABLE _settings AS

SELECT c.stay_id,

  COUNT(*) FILTER(WHERE rc.respchartoffset BETWEEN c.t0-60 AND c.t0) vent_setting_rows_1h,

  COUNT(*) FILTER(WHERE rc.respchartoffset BETWEEN c.t0-120 AND c.t0) vent_setting_rows_2h,

  COUNT(*) FILTER(WHERE rc.respchartoffset BETWEEN c.t0-240 AND c.t0) vent_setting_rows_4h,

  COUNT(*) FILTER(WHERE rc.respchartoffset BETWEEN c.t0-360 AND c.t0) vent_setting_rows_6h

FROM _candidate c LEFT JOIN respiratorycharting rc ON rc.patientunitstayid=c.stay_id

  AND rc.respchartvaluelabel IN ('PEEP','Tidal Volume (set)','Plateau Pressure','Mechanical Ventilator Mode')

  AND rc.respchartoffset BETWEEN c.t0-360 AND c.t0

GROUP BY c.stay_id;


SELECT c.stay_id,c.hospitalid,c.t0,c.treatment,

       COALESCE(a.strict_interval,false)::int strict_interval,

       COALESCE(a.airway_6h,false)::int airway_6h,

       COALESCE(a.airway_12h,false)::int airway_12h,

       COALESCE(a.airway_24h,false)::int airway_24h,

       a.last_airway_24h,

       CASE WHEN a.last_airway_offset_24h IS NOT NULL THEN (c.t0-a.last_airway_offset_24h)/60.0 END hours_since_airway,

       s.vent_setting_rows_1h,s.vent_setting_rows_2h,s.vent_setting_rows_4h,s.vent_setting_rows_6h,

       (COALESCE(a.airway_12h,false) AND s.vent_setting_rows_2h>0)::int pragmatic_12h_airway_2h_setting,

       (COALESCE(a.airway_24h,false) AND s.vent_setting_rows_4h>0)::int pragmatic_24h_airway_4h_setting

FROM _candidate c LEFT JOIN _airway a USING(stay_id) LEFT JOIN _settings s USING(stay_id)

ORDER BY c.stay_id;

