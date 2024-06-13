#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/nietoff/tfg/src/turtlebot3_autorace-kinetic-devel/turtlebot3_autorace_core"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/nietoff/tfg/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/nietoff/tfg/install/lib/python3/dist-packages:/home/nietoff/tfg/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/nietoff/tfg/build" \
    "/usr/bin/python3" \
    "/home/nietoff/tfg/src/turtlebot3_autorace-kinetic-devel/turtlebot3_autorace_core/setup.py" \
     \
    build --build-base "/home/nietoff/tfg/build/turtlebot3_autorace-kinetic-devel/turtlebot3_autorace_core" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/nietoff/tfg/install" --install-scripts="/home/nietoff/tfg/install/bin"
