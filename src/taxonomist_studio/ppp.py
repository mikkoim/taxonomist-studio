import pyautogui as ag
import PySimpleGUI as sg
from win32gui import GetWindowText, GetForegroundWindow
import string

CLICK_POSITION=(170,130)
CLICK_TEXT = "Click position"
button_well_list = [(str(row)+str(col)) for row in string.ascii_uppercase[:8] for col in range(1,13)]
negative_controls = ['A5', 'B4', 'B7', 'C9', 'D11', 'E2', 'E8', 'F3', 'F12', 'G1', 'H6', 'H10']

def paste_SampleName(name, click_position):
    ag.doubleClick(click_position[0],click_position[1])
    ag.press('del')
    ag.write(name, interval=0)

def open_position_options_window():
    global CLICK_POSITION, CLICK_TEXT
    layout_options = [[sg.T(key="-MOUSE_POSITION-"), sg.Input(key="-CLICK_POSITION-")], 
                    [sg.T(size=(50,2)), sg.B("Set click position in form <x,y>")]]
    window = sg.Window("Setting Click position", layout_options, modal=True, keep_on_top=True)
    while True:
        event, values = window.read(timeout=10)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        # Sets the click position 
        if event == "Set click position in form <x,y>":
            try: # Check validity of input
                position_text = values["-CLICK_POSITION-"].split(",")
                CLICK_POSITION = (int(position_text[0]), int(position_text[1]))
            except ValueError:
                CLICK_POSITION = (170,130)
                CLICK_TEXT = "Invalid click position"

            if (len(position_text) != 2):
                CLICK_TEXT = "Invalid click position"
            else:
                CLICK_TEXT = "Click position"
        
        # Update mouse position information
        pos = ag.position()
        window["-MOUSE_POSITION-"].update(f"Mouse: {pos.x},{pos.y} | {CLICK_TEXT}: {CLICK_POSITION[0], CLICK_POSITION[1]}")
        
    window.close()

def open_negative_controls_window():
    global negative_controls
    layout_options = [[sg.T(key='-OUTPUT_TXT-')], 
                    [sg.T('Enter well name list:'), sg.Input(key='-NEGATIVE_CONTROL_TXT-')], 
                    [sg.T(size=(50,2)), sg.B('Set negative controls')]]
    window = sg.Window('Setting negative controls', layout_options, modal=True, keep_on_top=True)

    while True:
        event, values = window.read(timeout=10)
        window['-OUTPUT_TXT-'].update('Current negative controls: '+ str(negative_controls))
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        # Sets the click position 
        if event == 'Set negative controls':
            try: # Check validity of input
                negative_controls = values['-NEGATIVE_CONTROL_TXT-'].split(',')
                nc_check = all(well in button_well_list for well in negative_controls)
                if nc_check == True:
                    window['-OUTPUT_TXT-'].update('New setting: '+ str(negative_controls))
                else:
                    window['-OUTPUT_TXT-'].update('Invalid input')
            except ValueError:
                negative_controls = ['A5', 'B4', 'B7', 'C9', 'D11', 'E2', 'E8', 'F3', 'F12', 'G1', 'H6', 'H10']
                window['-OUTPUT_TXT-'].update('Invalid input')
        
    window.close()

def run_ppp(args):
    sg.theme('DarkGreen')

    layout = [  [sg.Menu([['Options', ['Set click position', 'Set negative controls', 'Hide/Unhide wells']]])],
                [sg.T('Automatically insert plate position to SRS')],
                [sg.T('Which plate are you processing right now?'), sg.Input(key='-INPUT-', size=(10,2)), sg.B('Ok', key='-OK-')],
                [sg.T(size=(35,2), key='-OUTPUT-'), sg.B('Next position', key='-POSITION-')],
                [sg.B(f"A{num}", visible=False) for num in range(1, 13)],
                [sg.B(f"B{num}", visible=False) for num in range(1, 13)],
                [sg.B(f"C{num}", visible=False) for num in range(1, 13)],
                [sg.B(f"D{num}", visible=False) for num in range(1, 13)],
                [sg.B(f"E{num}", visible=False) for num in range(1, 13)],
                [sg.B(f"F{num}", visible=False) for num in range(1, 13)],
                [sg.B(f"G{num}", visible=False) for num in range(1, 13)],
                [sg.B(f"H{num}", visible=False) for num in range(1, 13)],
                [sg.B('Quit')]]

    # Create the window
    window = sg.Window('Plate Position Paster', layout, keep_on_top=True)

    # Create the options (click position) window


    toggle = True
    CURRENT_WELL = 0
    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read(timeout=10)
        # See if user wants to quit or window was closed
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break
        # Set options
        if event == 'Set click position':
            open_position_options_window()
        if event == 'Hide/Unhide wells':
            for button in button_well_list:
                window[button].update(visible=toggle)
            toggle = not toggle
        if event == 'Set negative controls':
            open_negative_controls_window()


        # Paste plate that is being processed and selected well to Sample Name in SRS
        if event in button_well_list:
            ag.click(1000,15)
            if GetWindowText(GetForegroundWindow()) == 'SRS':
                if event in negative_controls:
                    paste_SampleName(values['-INPUT-']+'_'+event+'=NEGATIVE_CONTROL', CLICK_POSITION)
                    window['-OUTPUT-'].update('Negative control: ' + values['-INPUT-']+'_'+event)
                    CURRENT_WELL = button_well_list.index(event) +1
                else:
                    paste_SampleName(values['-INPUT-']+'_'+event, CLICK_POSITION)
                    window['-OUTPUT-'].update('Pasted ' + values['-INPUT-']+'_'+event)
                    CURRENT_WELL = button_well_list.index(event) +1
            else:
                window['-OUTPUT-'].update('SRS window could not be selected')
        if event == '-POSITION-':
            ag.click(1000,15)
            if GetWindowText(GetForegroundWindow()) == 'SRS':
                if CURRENT_WELL < len(button_well_list):
                    if button_well_list[CURRENT_WELL] in negative_controls:
                        paste_SampleName(values['-INPUT-']+'_'+button_well_list[CURRENT_WELL]+'=NEGATIVE_CONTROL', CLICK_POSITION)
                        window['-OUTPUT-'].update('Negative control: ' + values['-INPUT-']+'_'+button_well_list[CURRENT_WELL])
                    else:
                        paste_SampleName(values['-INPUT-']+'_'+button_well_list[CURRENT_WELL], CLICK_POSITION)
                        window['-OUTPUT-'].update('Pasted ' + values['-INPUT-']+'_'+button_well_list[CURRENT_WELL])
                    CURRENT_WELL += 1
                else:
                    window['-OUTPUT-'].update('Pasted all wells. Starting from A1.')
                    CURRENT_WELL = 0
            else:
                window['-OUTPUT-'].update('SRS window could not be selected')

        # Output a message to the window
        if event == '-OK-':
            if values['-INPUT-'] == '':
                window['-OUTPUT-'].update('No name provided')
            else:
                window['-OUTPUT-'].update('Processing plate ' + values['-INPUT-'] + ":")


    # Finish up by removing from the screen
    window.close()