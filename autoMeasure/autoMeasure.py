import sys
sys.path.insert(0, "../SIFTMatching")
sys.path.insert(0, "../MeasureCP")
import numpy as np
from measureCP import getIO
from pixel2fiducial import allDist
from scipy import spatial
from SIFTMatching import match


def getTree(data, sf):
    """Initialize the point cloud KD-tree."""
    X = -data["Y"]  # Use the Y coordinates as image column
    Y = -data["Z"]  # Use the Z coordinates as image row

    # Get image resolution
    width = int((X.max() - X.min()) * sf)
    height = int((Y.max() - Y.min()) * sf)

    # Generate grid points
    rangeX = np.linspace(X.min(), X.max(), num=width)
    rangeY = np.linspace(Y.min(), Y.max(), num=height)

    # Create KD tree for point query
    tree = spatial.cKDTree(zip(X, Y))

    return tree, rangeX, rangeY


def main():
    # Define input file names
    LiDARFileName = '../ptCloud/XYZ.xyz'
    IOFileName = '../param/IO.txt'
    sf = 50     # Scale factor for LiDAR image
    LiDARImgFileName = '../images/P1_L_RGB50.png'
    photoName = '../images/P2_R.jpg'
    outputFileName = 'result.txt'

    # SIFT matching parameters
    ratio = 0.8
    showResult = True

    # Read LiDAR points information
    data = np.genfromtxt(
        LiDARFileName,
        dtype=[('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')],
        skip_header=1,
        usecols=(0, 1, 2))

    # Read interior orientation parameter
    IO = getIO(IOFileName)

    # Get the KD-tree object and grid point coordinates
    tree, rangeX, rangeY = getTree(data, sf)

    # Perform SIFT matching with LiDAR image and photo
    LiDARPt, photoPt = map(lambda x: x.reshape(-1, 2), match(
        LiDARImgFileName,
        photoName,
        ratio,
        show=showResult))

    LiDARPt = LiDARPt.astype(int)

    # Generate observation file for space resection
    ptSet = []
    for i in range(LiDARPt.shape[0]):
        # Search the index of object point
        dis, loc = tree.query(
            [rangeX[LiDARPt[i, 0]], rangeY[LiDARPt[i, 1]]],
            k=1, distance_upper_bound=0.01)

        # The case where the object point can be found
        if loc < len(data):
            imgPt = allDist(photoPt[i, 1], photoPt[i, 0], IO)
            objPt = (data['X'][loc], data['Y'][loc], data['Z'][loc])
            ptSet.append(imgPt + objPt)

    ptSet = np.array(ptSet)

    # Write out the result
    np.savetxt(
        outputFileName,
        ptSet,
        fmt="Pt %.6f %.6f %.6f %.6f %.6f" + " 0.005" * 3,
        header=str(IO['f']),
        comments='')

if __name__ == '__main__':
    main()
