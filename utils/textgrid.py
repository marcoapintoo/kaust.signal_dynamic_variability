"""
.TextGrid looks like a YAML file. Therefore we can open it using a specific library, or processing the file by ourselves.
In my case, I prefer to transform the file by-hand, because the output won't be a dict, but a NUmpy array. In any case, some post-processing will be need.
"""
import numpy as np
from .sliding_window import sliding_window

class TextGridExtractor:
    def __init__(self, word_sampling_frequency, precision=3):
        """
        word_sampling_frequency: 1/period where we will count the most common word.
        precision: number of digits of the movie_sampling_frequency
        """
        #movie_sampling_frequency: points sampled from the info in the file.
        self.script_values = []
        self.codewords = {}
        self.word_sampling_period = 1/word_sampling_frequency # in [secs]
        self.precision = precision

    @property
    def words_to_codes(self):
        return {v: k for k, v in self.codewords.items()}

    @property
    def categories(self):
        return [v[0] for v in self.script_values]

    def words_in_category(self, category, return_code=False):
        category_index = self.categories.index(category)
        words_to_codes = self.words_to_codes
        if return_code:
            return self.script_values[category_index][-1].astype("i")
        return [words_to_codes[v] for v in self.script_values[category_index][-1]]

    @property
    def sequences_by_category_word(self):
        words_to_codes = self.words_to_codes
        word_marks = {}
        for name, vect in self.script_values:
            for m in np.unique(vect).astype("i"):
                key = "{0}: {1}".format(name, words_to_codes[m])
                mark = (vect == m).astype("i2")
                word_marks[key] = mark
        return word_marks

    def extract_from_file(self, filename, maximum_length=None):
        """
        maximum_length: this function assume that there is a undetermined delay at the start of the file. This parameter is used to determine that delay length.
        """
        movie_sampling_frequency = 10 ** self.precision # in [Hz] or [1/secs]
        M = int(movie_sampling_frequency * self.word_sampling_period)
        script_keys = {d: i for i, (d, v) in enumerate(self.script_values)}
        current_values = np.array([]) if len(self.script_values) == 0 else np.zeros(len(self.script_values[0][1]), dtype="i")
        time_delay = 0
        maximum_time = None
        type_sources = None
        is_item = lambda x: "item" in x
        is_interval = lambda x: "interval" in x
        get_key = lambda x: [m.strip() for m in x.split("=")] if "=" in x else ["", x.strip()]
        with open(filename, "rt") as f:
            line = f.readline()
            while line != "":
                key, val = get_key(line)
                if key == "xmax":
                    maximum_time = float(val)
                    maximum_length = maximum_length if maximum_length is not None else maximum_time
                    time_delay = maximum_time - maximum_length
                    #print("time_delay:", time_delay)
                elif key == "size":
                    type_sources = int(val)
                line = f.readline()
                if "item []" in line:
                    break
            for m in range(1, type_sources + 1):
                interval_size, initial_time, maximum_time, name = None, None, None, None
                #print(line)
                while "intervals [" not in line:
                    key, val = get_key(line)
                    #print(key, '=>', val, 0, "size" in key)
                    if "size" in key:
                        interval_size = int(val)
                        #print(key, val, interval_size, -1)
                    elif "xmin" in key:
                        initial_time = np.round(float(val), self.precision)
                        #print(key, val, interval_size, -1)
                    elif "xmax" in key:
                        maximum_time = np.round(float(val), self.precision)
                        #print(key, val, interval_size, -1)
                    elif "name" in key:
                        name = eval(val)
                        #print(key, val, interval_size, -1)
                    line = f.readline()
                #print(interval_size, line, 0)
                #print(interval_size, initial_time, maximum_time, name, line)
                vals = []
                for i in range(1, 1 + interval_size):
                    line = f.readline()
                    t0, t1, label = None, None, None
                    for _ in range(3):
                    #while "intervals [" not in line and "item [" not in line:
                        key, val = get_key(line)
                        #print(key, val)
                        if key == "xmin":
                            t0 = float(val)
                            t0 = max(0, t0 - time_delay)
                            #print(val, t0)
                        elif key == "xmax":
                            t1 = float(val)
                            t1 = max(0, t1 - time_delay)
                        elif key == "text":
                            try:
                                label = eval(val)
                            except:
                                #print(val, key, name, i)
                                line2 = f.readline()
                                key, val = get_key((line + line2).replace("\n", ""))
                                #print(val, key, name, i)
                                pass
                            label = eval(val).strip() #Ugly!s
                            self.codewords.setdefault(label, len(self.codewords))
                            label = self.codewords[label]
                        line = f.readline()
                    m = round((t1 - t0) * movie_sampling_frequency)
                    #m = int((t1 - t0) * sampling_frequency)
                    #print(t0, t1, m)
                    vals.append(np.ones(m) * label)
                vals = np.concatenate(vals).astype("i")
                #print("::", len(script[-1][-1]) / sampling_frequency, sliding_window(script[-1][-1], size=M, stepsize=M).shape)
                vals = np.apply_along_axis(
                    lambda x: np.argmax(np.bincount(x)), 
                    1, 
                    sliding_window(vals, size=M, stepsize=M))
                #print("  ::", script[-1][-1].shape)
                if name in script_keys:
                    self.script_values[script_keys[name]][1] = np.concatenate([self.script_values[script_keys[name]][1], vals])
                else:
                    self.script_values.append([name, np.concatenate([current_values, vals])])

