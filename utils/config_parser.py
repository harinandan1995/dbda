import os
import yaml

class Config:
    """
    Config parser to get parameters from yaml config_parser

    :param file_path: Path to the yaml config file
    """

    def __init__(self, file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError('Invalid config path provided.')

        with open(file_path, 'r') as stream:
            try:
                self.data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc

    def get_string(self, key, default_value=None):

        """
        Used to get a string parameter from the config

        :param key: Value of this key will be returned from the config
        :param default_value: Default value if the given key is not found in the config
        """

        if key not in self.data and default_value is None:
            raise AttributeError('%s not found in the config and not default value provided' % key)
        else:
            try:

                if key in self.data:
                    if self.data[key] == '':
                        return None
                    return str(self.data[key])
            except e:
                raise ValueError('Cannot convert type %s to string' % type(self.data[key]))

            try:
                return str(default_value)
            except e:
                raise ValueError('Cannot convert type %s to string' % type(default_value))

    def get_float(self, key, default_value=None):

        """
        Used to get a float parameter from the config

        :param key: Value of this key will be returned from the config
        :param default_value: Default value if the given key is not found in the config
        """

        if key not in self.data and default_value is None:
            raise AttributeError('%s not found in the config and not default value provided' % key)
        else:
            try:
                if key in self.data:
                    return float(self.data[key])
            except e:
                raise ValueError('Cannot convert type %s to float' % type(self.data[key]))

            try:
                return float(default_value)
            except e:
                raise ValueError('Cannot convert type %s to float' % type(default_value))

    def get_int(self, key, default_value=None):

        """
        Used to get a int parameter from the config

        :param key: Value of this key will be returned from the config
        :param default_value: Default value if the given key is not found in the config
        """

        if key not in self.data and default_value is None:
            raise AttributeError('%s not found in the config and not default value provided' % key)
        else:
            try:
                if key in self.data:
                    return int(self.data[key])
            except e:
                raise ValueError('Cannot convert type %s to int' % type(self.data[key]))

            try:
                return str(default_value)
            except e:
                raise ValueError('Cannot convert type %s to int' % type(default_value))
