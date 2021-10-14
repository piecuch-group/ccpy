
def FileCheck(filename):
    try:
      open(filename, "r")
      return True
    except IOError:
      print('Error: {} does not appear to exist.'.format(filename))
      return False