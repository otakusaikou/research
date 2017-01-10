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

--Create sequence for table 'colorinfo'
CREATE SEQUENCE public.colorinfo_id_seq START 1;

/*
Create table 'colorinfo', primary key is 'id'
Foreign keys are point3d_no, referenced from point3d(id)
                 image_no, referenced from image(id)*/
CREATE TABLE public.colorinfo
(
    id integer NOT NULL DEFAULT nextval('colorinfo_id_seq'::regclass),
    r integer,
    g integer,
    b integer,
    row float8,
    col float8,
    point3d_no integer,
    image_no integer,
    CONSTRAINT COLORINFOPK PRIMARY KEY (id),
    CONSTRAINT COLORINFOFK_POINT3D
        FOREIGN KEY (point3d_no) REFERENCES point3d(id),
    CONSTRAINT COLORINFOFK_IMAGE
        FOREIGN KEY (image_no) REFERENCES image(id)
);
