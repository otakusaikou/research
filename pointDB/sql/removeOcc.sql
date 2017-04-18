--Create table to store rows having its id not in the tmpid table
--Reference: https://goo.gl/BbhNZv
CREATE TABLE colorinfo_tmp AS
SELECT C.*
FROM colorinfo C
LEFT JOIN tmpid D USING (id)
WHERE D.id IS NULL;

--DROP the whole colorinfo table which contains the id of the occluded points
DROP TABLE colorinfo;

--Rename the template table and add PK, FK constraints and default id value
ALTER TABLE colorinfo_tmp RENAME TO colorinfo;
ALTER TABLE colorinfo ADD CONSTRAINT COLORINFOPK PRIMARY KEY (id);
ALTER TABLE colorinfo ADD CONSTRAINT COLORINFOFK_POINT3D FOREIGN KEY (point3d_no) REFERENCES point3d(id);
ALTER TABLE colorinfo ADD CONSTRAINT COLORINFOFK_IMAGE FOREIGN KEY (image_no) REFERENCES image(id);
ALTER TABLE colorinfo ALTER COLUMN id SET DEFAULT nextval('colorinfo_id_seq'::regclass);

--DROP the occluded point id table
DROP TABLE IF EXISTS tmpid
