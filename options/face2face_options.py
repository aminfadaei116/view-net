import argparse


class Face2FaceOption:
    """
    This class defines options used for the face2face model
    """
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description='Gather the parameters for the project')
        parser.add_argument('--used_device', type=str, default='none',
                            help='Which device are you using at the moment-> '
                                 '[ YorkU | Ubisoft | laptop ]')
        self.parser = parser

    def get_parser(self):
        """
        get the parser parameters
        """
        return self.parser.parse_args()