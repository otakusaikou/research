#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import psycopg2


conn = None


def mousePress(event):
    """Handle the mouse press event."""
    if event.inaxes:    # Check if the mouse is in the image area
        ax = event.inaxes

        # If the event is mouse right click, clean the canvas
        if event.button == 3 and ax.is_first_col():
            clean(ax.figure.get_axes())

        # Draw the upper-left corner of the rectangle when user hold the
        # control key
        elif event.button == 1 and event.key == "control":
            clean(ax.figure.get_axes())
            if ax.is_first_col():
                ax.plot(event.xdata, event.ydata, "ro")
        ax.figure.canvas.draw_idle()    # Update the canvas


def mouseRelease(event):
    """Handle the mouse release event."""
    if event.inaxes:    # Check if the mouse is in the image area
        ax = event.inaxes

        # Draw the other corners of the rectangle when user hold the
        # control key
        if event.button == 1 and event.key == "control":
            if len(ax.lines) == 4:  # Clean old points and lines
                map(lambda e: e.remove(), ax.lines[:3])
            ax.plot([event.xdata, event.xdata, ax.lines[0]._x[0]],
                    [ax.lines[0]._y[0], event.ydata, event.ydata], "ro")

            ax.plot(
                [ax.lines[0]._x[0], event.xdata,
                    event.xdata, ax.lines[0]._x[0], ax.lines[0]._x[0]],
                [ax.lines[0]._y[0], ax.lines[0]._y[0],
                    event.ydata, event.ydata, ax.lines[0]._y[0]], "r-")

            ax.figure.canvas.draw_idle()    # Update the canvas

            cur = conn.cursor()     # Get cursor object of database connection

            col = ax.lines[-1].get_xdata()
            row = ax.lines[-1].get_ydata()
            with open('sql/queryByRowCol.sql') as sql:
                cur.execute(sql.read(), (
                    ax.get_title(),
                    row.min(), row.max(), col.min(), col.max()))

            objPts = np.array(cur.fetchall())
            X, Y, Z, R, G, B = np.hsplit(objPts, 6)

            ax3d = ax.figure.get_axes()[1]
            ax3d.scatter(X, Y, Z, c='b')

            # Make the 3d plot equal scale
            plotRadius = max(map(lambda x: x.max() - x.min(), [X, Y, Z])) / 2.0

            midX = (X.max()+X.min()) * 0.5
            midY = (Y.max()+Y.min()) * 0.5
            midZ = (Z.max()+Z.min()) * 0.5
            ax3d.set_xlim(midX - plotRadius, midX + plotRadius)
            ax3d.set_ylim(midY - plotRadius, midY + plotRadius)
            ax3d.set_zlim(midZ - plotRadius, midZ + plotRadius)

            # Output the result
            np.savetxt(
                'result.txt',
                objPts,
                fmt="%.6f %.6f %.6f %d %d %d",
                header="X Y Z R G B",
                comments='')

            # Free the memory
            del objPts, X, Y, Z, R, G, B, cur
            gc.collect()


def clean(axes):
    """Clean all the objects in given axes."""
    while axes[0].lines:
        axes[0].lines[0].remove()
    axes[1].cla()


def dispGUI(imgFileName, conn):
    """Display the image and 3d plot."""
    # Read image and convert it from BGR to RGB
    img = cv2.cvtColor(cv2.imread(imgFileName), cv2.COLOR_BGR2RGB)

    # Create figure
    fig = plt.figure("Point cloud explorer", figsize=(17, 6))
    fig.canvas.draw_idle()

    # Show the rgb image
    ax = fig.add_subplot(121)
    plt.title("%s" % os.path.split(imgFileName)[-1], size=15)
    ax.axis([0, img.shape[1], img.shape[0], 0])
    ax.axis("off")
    ax.imshow(img, interpolation="none")

    # Plot 3d plot for point cloud
    fig.add_subplot(122, projection='3d')

    fig.tight_layout(pad=3, w_pad=5)

    # Bind the mouse event
    fig.canvas.mpl_connect("button_press_event", mousePress)
    fig.canvas.mpl_connect("button_release_event", mouseRelease)

    plt.show()


def main():
    # Define database connection parameters
    host = 'localhost'
    port = '5432'
    dbName = 'pointdb'
    user = 'postgres'

    # Define imae file name
    imgFileName = "../images/P1_L.jpg"

    # Connect to database
    try:
        global conn
        conn = psycopg2.connect(
            "dbname='%s' user='%s' host='%s' port='%s'" %
            (dbName, user, host, port))
    except psycopg2.OperationalError:
        print "Unable to connect to the database."
        return -1

    dispGUI(imgFileName, conn)
    conn.close()

    return 0

if __name__ == '__main__':
    main()
