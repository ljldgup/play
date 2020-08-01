import win32gui


def _MyCallback(hwnd, extra):
    windows = extra
    temp = []
    temp.append(hex(hwnd))
    temp.append(win32gui.GetClassName(hwnd))
    temp.append(win32gui.GetWindowText(hwnd))
    windows[hwnd] = temp


def TestEnumWindows():
    windows = {}
    win32gui.EnumWindows(_MyCallback, windows)
    print("Enumerated a total of  windows with %d classes", (len(windows)))
    print('------------------------------')

    # print classes
    print('-------------------------------')

    for item in windows:
        print(windows[item])

    print("Enumerating all windows...")