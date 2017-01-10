--Create sequence for table 'point3d'
CREATE SEQUENCE public.point3d_id_seq START 1;

--Create table 'point3d', primary key is 'id'
CREATE TABLE public.point3d
(
  id integer NOT NULL DEFAULT nextval('point3d_id_seq'::regclass),
  x float8,
  y float8,
  z float8,
  I integer,
  CONSTRAINT POINT3DPK PRIMARY KEY (id)
);

--Create sequence for table 'image'
CREATE SEQUENCE public.image_id_seq START 1;

--Create table 'image', primary key is 'id'
CREATE TABLE public.image
(
  id integer NOT NULL DEFAULT nextval('image_id_seq'::regclass),
  name varchar(20),
  omega float8,
  phi float8,
  kappa float8,
  xl float8,
  yl float8,
  zl float8,
  CONSTRAINT IMAGEPK PRIMARY KEY (id)
);

--Create sequence for table 'point2d'
CREATE SEQUENCE public.point2d_id_seq START 1;

/*
Create table 'point2d', primary key is 'id'
Foreign key is image_no, referenced from image(id)*/
CREATE TABLE public.point2d
(
  id integer NOT NULL DEFAULT nextval('point2d_id_seq'::regclass),
  row float8,
  col float8,
  image_no integer,
  CONSTRAINT POINT2DPK PRIMARY KEY (id),
  CONSTRAINT POINT2DFK_IMAGE
    FOREIGN KEY (image_no) REFERENCES image(id)
);

--Create sequence for table 'color'
CREATE SEQUENCE public.color_id_seq START 1;

/*
Create table 'color', primary key is 'id'
Foreign keys are point3d_no, referenced from point3d(id)
                 point2d_no, referenced from point2d(id)*/
CREATE TABLE public.color
(
  id integer NOT NULL DEFAULT nextval('color_id_seq'::regclass),
  r integer,
  g integer,
  b integer,
  point3d_no integer,
  point2d_no integer,
  CONSTRAINT COLORPK PRIMARY KEY (id),
  CONSTRAINT COLORFK_POINT3D
    FOREIGN KEY (point3d_no) REFERENCES point3d(id),
  CONSTRAINT COLORFK_POINT2D
    FOREIGN KEY (point2d_no) REFERENCES point2d(id)
);
