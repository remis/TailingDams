def read_label_dictionary(file_path):
    f = open(file_path)
    output_dictionary = dict()
    for line in f.readlines():
        words = line.split()
        code = int(words[0][:-1])
        label = words[1]
        output_dictionary[code] = label

    return output_dictionary
