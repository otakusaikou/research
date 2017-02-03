SELECT cid, x, y, z, r, g, b, row, col
FROM point3d P3D JOIN (SELECT id cid, r, g, b, row, col, point3d_no
                       FROM colorinfo C
                       WHERE C.image_no = (SELECT id I_id
                                           FROM image
                                           WHERE position(%s in name) != 0)) C2 ON (P3D.id = C2.point3d_no);
