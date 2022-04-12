#!/usr/bin/env python3
###
#
# CIS Top of Atmosphere Radiance Calibration
#
# Program Description : This file creates the directory structure for the Landsat-Buoy-Calibration
#                           program and launches either the terminal or gui version of the program
#                           based on user selection and terminal interface capabilities
# Created By          : Benjamin Kleynhans
# Creation Date       : June 20, 2019
# Authors             : Benjamin Kleynhans
#
# Last Modified By    : Benjamin Kleynhans
# Last Modified Date  : August 1, 2019
# Filename            : cis-tarca.py
#
###

# Imports
import sys, os, inspect, pdb
import time
import subprocess as sp
import ctypes

# Erases a line of output without adding a newline
ERASE_LINE = '\x1b[2K'

PROJECT_ROOT = ''
GUI_ROOT = os.path.join(PROJECT_ROOT, 'gui/')

# Calculate fully qualified path to location of program execution
def get_module_path():

    filename = inspect.getfile(inspect.currentframe())
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    return path, filename


# # Create any folders that are required and not currently available
# def check_required_directories():

#     from tools import test_paths

#     required_directories = {
#         'input',
#         'input/batches',
#         'output',
#         'output/processed_images',
#         'output/single',
#         'output/single/sc',
#         'output/single/sc/buoy',
#         'output/single/sc/toa',
#         'output/single/sw',
#         'output/single/sw/lst',
#         'output/batch',
#         'output/batch/data',
#         'output/batch/data/sc',
#         'output/batch/data/sc/buoy',
#         'output/batch/data/sc/toa',
#         'output/batch/data/sw',
#         'output/batch/data/sw/lst',
#         'output/batch/graphs',
#         'logs',
#         'logs/status',
#         'logs/status/single',
#         'logs/status/batch',
#         'logs/output',
#         'logs/output/single',
#         'logs/output/batch'
#     }

#     # Loop through the list of required directories and create any that don't exist
#     for directory in required_directories:
#         if not test_paths.main([directory, "-tdirectory"]):
#             test_paths.createDirectory(directory)


# Set environment variables to locate current execution path
def set_path_variables():

    # check_required_directories()

    path, filename = get_module_path()

    sys.path.append(path)
    sys.path.append(path + "/gui")
    # sys.path.append(path + "/buoycalib/")
    # sys.path.append(path + "/downloaded_data")
    # sys.path.append(path + "/input")
    # sys.path.append(path + "/logs")
    # sys.path.append(path + "/modules")
    # sys.path.append(path + "/modules/db")
    # sys.path.append(path + "/modules/gui")
    # sys.path.append(path + "/modules/stp")
    # sys.path.append(path + "/modules/core")
    # sys.path.append(path + "/output")
    # sys.path.append(path + "/tools")


# # Tests whether the online data sources are available
# def source_test(address, missing_sources):

#     import test_paths

#     if not test_paths.main([address, '-tserver']):

#         sys.stdout.write(" --> NOT AVAILABLE!!!")
#         sys.stdout.flush()

#         missing_sources.append(address)
#     else:

#         sys.stdout.write(" >>> available.")
#         sys.stdout.flush()

#         missing_sources = None

#     time.sleep(0.5)

#     return missing_sources


# # Test if all data sources specified in buoycalib/settings.py are present
# def check_sources():

#     import settings

#     sources_available = True

#     sources = []

#     sources.append(settings.MERRA_SERVER)
#     sources.append(settings.NARR_SERVER)
#     sources.append(settings.NOAA_SERVER)
#     sources.append(settings.LANDSAT_S3_SERVER)
#     sources.append(settings.LANDSAT_EE_SERVER)

#     if settings.USE_MYSQL:
#         sources.append(settings.SQL_SERVER)

#     missing_sources = []

#     for source in sources:
#         sys.stdout.write(ERASE_LINE)
#         sys.stdout.write('\r Testing >>> ' + source)
#         source = source_test(source, missing_sources)

#     if (len(missing_sources)):
#         sources_available = False

#     return sources_available, missing_sources

# Test if the terminal session allows export of display
def export_display_available():
    
    returnValue = False
    
    if "DISPLAY" in os.environ:
        returnValue = True
    
    return returnValue


def parseArgs(args):

    import argparse

    parser = argparse.ArgumentParser(description='Calculate and compare the Top of Atmosphere Radiance values '
                                     'of a LandSat or Merra image using the modelled theoretical composition '
                                     'of atmospheric data.  NOAA buoy recorded data is used along with '
                                     'projections with MODTRAN.  Other features include split window.')

    parser.add_argument('-i', '--interface', default='gui', choices=['gui', 'terminal'], help='Choose if you want to use the graphical or terminal based interface.')

    return parser.parse_args(args)


def main(args):

    X11 = ctypes.CDLL('libX11.so')
    X11.XInitThreads()

    PROJECT_ROOT = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    sp.call('clear', shell = True)
    
    set_path_variables()
    
    from gui import tarca_gui
    tarca_gui.main(PROJECT_ROOT)

    # print()
    # print(" Please be patient while we test if the required data sources are available")
    # print()

    # data_sources_available, missing_sources = check_sources()

    # if (data_sources_available):

    #     sys.stdout.write(ERASE_LINE)
    #     sys.stdout.write("\r     --> All data sources are accounted for <--")
    #     print()

    #     from modules.core import menu
    #     from modules.gui import tarca_gui

    #     launch = parseArgs(args)

    #     if launch.interface == 'gui':
    #         if export_display_available():
    #             #print("\n !!! the GUI has not yet been implemented, launching text interface !!!")
    #             #menu.main(PROJECT_ROOT)
    #             tarca_gui.main(PROJECT_ROOT)
    #         else:
    #             print("\n !!! Your terminal session does not support X-window graphics !!!" \
    #                   "\n !!! Press Enter to start the terminal version of the program !!!")
                
    #             input("\n\n  Press Enter to continue...")
                
    #             menu.main(PROJECT_ROOT)
    #     else:
    #         menu.main(PROJECT_ROOT)

    # else:        
    #     print("\n\n!!!")
    #     print("!")
    #     print("! The program has quit because the following data source/s are currently " \
    #             "not available !")
    #     print("!")
    #     print("!!!")
    #     print("!")

    #     for source in missing_sources:
    #         print("!    >>  {}".format(source))

    #     print("!")
    #     print("!!!\n")

        
    # sp.call('clear', shell = True)


if __name__ == '__main__':

    args = main(sys.argv[1:])
