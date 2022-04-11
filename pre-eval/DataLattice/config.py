
def read_config():
    ConfigFilePath = 'config'
    config_info = {}
    with open(ConfigFilePath) as file:
        for line in file:
            line = line.rstrip('\n')
            line = line.rstrip('\r\n')
            item = line.split('=')
            key = item[0]
            if key == 'txtPath':
                config_info[key] = item[1]
            elif key == 'output':
                config_info[key] = item[1]
            elif key == 'startTime':
                config_info[key] = item[1]
            elif key == 'endTime':
                config_info[key] = item[1]
            elif key == 'lonBegin':
                config_info[key] = float(item[1])
            elif key == 'lonEnd':
                config_info[key] = float(item[1])
            elif key == 'latBegin':
                config_info[key] = float(item[1])
            elif key == 'latEnd':
                config_info[key] = float(item[1])
            elif key == 'timeGap':
                config_info[key] = int(item[1])
            elif key == 'TruthFileDir':
                config_info[key] = item[1]
            else:
                print('no this item: {}'.format(key))
                assert False
    return config_info


if __name__ == "__main__":
    t = read_config()
    print(t)



