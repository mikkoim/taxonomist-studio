import pyautogui as ag
import PySimpleGUI as sg
import time
import pandas as pd
from win32gui import GetWindowText, GetForegroundWindow
import logging

logger = logging.getLogger(__name__)

def paste_txt(text, x, y):
    ag.doubleClick(x,y)
    ag.press('del')
    ag.write(str(text).replace(' ', '_'), interval=0)

def open_extended():
    ag.click(150,420)
    time.sleep(0.25)
    ag.click(150,450)

def paste_MetaData(SampleName, SpeciesName, OtherInfo, CollectionDate, Station, Location, Lat, Long, Determiner, Sex):
    paste_txt(SampleName, 170,130)
    paste_txt(SpeciesName, 30,230)
    paste_txt(OtherInfo, 30,280)
    open_extended()
    paste_txt(CollectionDate, 30,470)
    paste_txt(Station, 30,520)
    paste_txt(Location, 30,570)
    paste_txt(Lat, 30,620)
    paste_txt(Long, 210,620)
    paste_txt(Determiner, 30,670)
    paste_txt(Sex, 30,720)

def paste_MetaData16(SampleName, SpeciesName, OtherInfo, CollectionDate, Station, Location, Lat, Long, Determiner, Sex):
    """Metadata pasting for SRS 1.6.0"""
    paste_txt(SampleName, 170,130)
    paste_txt(SpeciesName, 30,230)
    paste_txt(CollectionDate, 30,275)
    paste_txt(Station, 30,320)
    paste_txt(Location, 30,360)
    paste_txt(Lat, 30,410)
    paste_txt(Long, 220,410)
    paste_txt(Determiner, 30,450)
    paste_txt(Sex, 30,500)
    paste_txt(OtherInfo, 30,540)

def run_mdma(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    sg.theme('DarkGreen')
    ag.PAUSE = 0.001
    
    layout = [  [sg.T('Automatically insert metadata to SRS')],
                [sg.T('Choose input file: '), sg.FileBrowse(key='-FILE-', file_types=(('xlsx', '*.xlsx'),)), sg.B('Read')],
                [sg.T(key='-LoadedFILE-'), sg.T(key='-FILEinfo-')],
                [sg.T('Start from excel sheet row'), sg.Input(key='-starting_line-'), sg.B('Ok')],
                [sg.T(' ')],
                [sg.T(size=(45,2), key='-WARNING-'), sg.B('Next Line')],
                [sg.B('Quit')]]

    # Create the window
    window = sg.Window('MetaDataManagingApplication', layout, keep_on_top=True)
    count = 0

    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        # See if user wants to quit or window was closed
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break
        # When pressing 'Read' button: Check if file has been selected. If yes, load excel file.
        if event == 'Read':
            if values['-FILE-'] == '':
                window['-LoadedFILE-'].update('Please select a file')
            else:
                loaded_data = pd.read_excel(values['-FILE-'], keep_default_na=False)
                window['-LoadedFILE-'].update('Loaded ' + str(*values['-FILE-'].split('/')[-1:]))
                window['-FILEinfo-'].update('containing ' + str(len(loaded_data.index)) + ' rows (without headline)')

        # When pressing 'Ok' button: Check if input was int. If yes, update starting row.
        if event == 'Ok':
            if values['-starting_line-'] == '':
                window['-WARNING-'].update('Missing Input')
            else:
                try: 
                    starting_line = int(values['-starting_line-'])
                    window['-WARNING-'].update('Starting from row: ' + str(starting_line))
                    count = starting_line - 2
                    if count <= -1:
                        window['-WARNING-'].update('Invalid line input: Headline cannot be pasted')
                    if count + 1 > len(loaded_data.index):
                        window['-WARNING-'].update('Invalid line input: File only has ' + str(len(loaded_data.index)+1) + ' rows')
                    else:
                        pass
                except Exception: 
                    window['-WARNING-'].update('Invalid line input')
                    logger.exception("Invalid line input")

        # When pressing 'Next Line' button: Check if file has been selected. If yes, check if SRS window can be selected. If yes, paste meta data.
        if event == 'Next Line':
            if values['-FILE-'] == '':
                window['-WARNING-'].update('No file selected')
            try: 
                loaded_data
                if count >= len(loaded_data.index):
                    window['-WARNING-'].update('End of excel file reached')
                else:
                    ag.click(1000,15)
                    logger.debug("Foreground window: %s" % GetForegroundWindow())
                    logger.debug("Window text: '%s'" % GetWindowText(GetForegroundWindow()))
                    if GetWindowText(GetForegroundWindow()) == 'SRS' \
                        or GetWindowText(GetForegroundWindow()) == 'SRS System':
                        window['-WARNING-'].update('pasting...')
                        try:
                            logging.debug("%s" % loaded_data.iloc[count])
                            if args.version == "1.6":
                                paste_MetaData16(*loaded_data.iloc[count])
                            else:
                                paste_MetaData(*loaded_data.iloc[count])
                            count +=1
                            window['-WARNING-'].update(f'Row {count+1} | Info succesfully pasted')
                        except Exception:
                            window['-WARNING-'].update("Error reading excel file. Check the contents.")
                            logging.exception("paste_MetaData error:")
                    else:
                        window['-WARNING-'].update('SRS window could not be selected')
            except Exception:
                window['-WARNING-'].update('Data not loaded. Read data first')
                logger.exception("Data not loaded.")

    # Finish up by removing from the screen
    window.close()