TEMPLATE = app
TARGET = multiplication
QT *= opengl xml
DEPENDPATH += .
INCLUDEPATH += /usr/local/qt-labs-opencl/src/opencl
LIBS += -L/usr/local/qt-labs-opencl/lib -lQtOpenCL -L/usr/lib/x86_64-linux-gnu -lOpenCL
QMAKE_CXXFLAGS += -Wno-write-strings
QMAKE_CXXFLAGS_DEBUG += -Wno-deprecated
QMAKE_CXXFLAGS_RELEASE += -Wno-deprecated

HEADERS +=
SOURCES += \
    main.cpp
