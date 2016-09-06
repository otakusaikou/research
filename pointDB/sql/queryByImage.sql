SELECT x, y, z, r, g, b, row, col
FROM (point3d P3D JOIN (SELECT r, g, b, row, col, point3d_no
                        FROM (color C JOIN (SELECT row, col, id P2DI_id
                                            FROM (point2d P2D JOIN (SELECT id I_id
                                                                    FROM image
                                                                    WHERE position(%s in name) != 0) I ON (P2D.image_no = I.I_id))) P2DI ON (C.point2d_no = P2DI.P2DI_id))) CP2DI ON (P3D.id = CP2DI.point3d_no)) P3DCP2DI;
