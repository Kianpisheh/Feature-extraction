from scipy.io import wavfile


def read_data(input_file):
    '''
    The interface for reading the unput files and segmentation times
    from a text file

    return:
        
    '''

    f = open(input_file, 'r')
    f_lines = f.readlines()
    
    return f_lines