"""It exposes utility functions for input parsing.
"""

def kv_str_to_tuple(value):
    """It splits the input key=value string into a tuple
    
    Arguments:
        value {string} -- A string value in the form of KEY=VALUE
    """
    value_pair = value.split('=')

    if len(value_pair) != 2:
        raise ValueError('Malformed input: {}'.format(value))

    raw_value = value_pair[1]
    parsed_value = None

    try:
        #Integer parsing
        parsed_value = int(raw_value)
    except ValueError:
        try:
            #Float parsing
            parsed_value = float(raw_value)
        except ValueError:
            #Boolean parsing
            raw_value_l = raw_value.lower()
            if raw_value_l == "true":
                parsed_value = True
            elif raw_value_l == "false":
                parsed_value = False
            else:
                #String
                parsed_value = raw_value

    return (value_pair[0], parsed_value)