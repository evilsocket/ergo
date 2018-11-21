from setuptools import setup, find_packages
from ergo.version import __version__

import os

long_description = 'Ergo utilities and training framework.'

setup( name                 = 'ergo',
       version              = __version__,
       description          = long_description,
       long_description     = long_description,
       author               = 'Simone Margaritelli',
       author_email         = 'evilsocket@gmail.com',
       url                  = 'http://www.github.com/evilsocket/ergo',
       packages             = find_packages(),
       scripts              = [ 'bin/ergo' ],
       license              = 'GPL',
       zip_safe             = False,
       classifiers          = [
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Intended Audience :: End Users/Desktop',
            'Intended Audience :: Developers',
            'Intended Audience :: System Administrators',
            'Intended Audience :: Information Technology',
            'License :: OSI Approved :: GNU General Public License (GPL)',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Unix',
            'Operating System :: POSIX',
            'Operating System :: Microsoft :: Windows',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Software Development :: Build Tools',
            'Topic :: Software Development :: Code Generators',
            'Topic :: Internet',
            'Natural Language :: English'
      ]
)
