import os
from typing import Dict, Any, Tuple

import pandas as pd


def splitpath(path: str) -> Tuple[str, str, str]:
    """ Splits path into: directory, filename, extension """
    directory, filename = os.path.split(path)
    basename, extension = os.path.splitext(filename)
    return directory, basename, extension


def dftodict(df: pd.DataFrame, key: str = 'image'
                       ) -> Dict[Any, Dict[Any, Any]]:
    """ Converts given DataFrame into a dictionary. """
    return {row.pop(key): row for row in df.to_dict('records')}
