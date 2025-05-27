from Pyro4 import expose


class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None, ngram_size=2):
        self.input_file = input_file_name
        self.output_file = output_file_name
        self.workers = workers
        self.ngram_size = ngram_size

    def solve(self):
        text = self.read_input()
        if not text:
            return

        words = text.split()
        if not words or len(words) < self.ngram_size:
            return


        ngrams = [' '.join(words[i:i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)]

        step = len(ngrams) // len(self.workers)
        mapped = []
        for i in range(len(self.workers)):
            chunk = ngrams[i * step: (i + 1) * step] if i < len(self.workers) - 1 else ngrams[i * step:]
            mapped.append(self.workers[i].mymap(chunk))

        reduced = self.myreduce(mapped)
        self.write_output(reduced)

    def preprocess_text(self, text):
        text = text.lower()
        cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        return cleaned

    @staticmethod
    @expose
    def mymap(ngrams):
        ngram_count = {}
        for ngram in ngrams:
            if ngram:
                ngram_count[ngram] = ngram_count.get(ngram, 0) + 1
        return ngram_count

    @staticmethod
    @expose
    def myreduce(mapped):
        final_count = {}
        for ngram_count in mapped:
            if hasattr(ngram_count, 'value'):
                ngram_count = ngram_count.value
            for ngram, count in ngram_count.items():
                final_count[ngram] = final_count.get(ngram, 0) + count
        return final_count

    def read_input(self):
        try:
            with open(self.input_file, 'r') as f:
                text = f.read().strip()
            return self.preprocess_text(text)
        except IOError:
            return ""

    def write_output(self, output):
        try:
            with open(self.output_file, 'w') as f:
                sorted_ngrams = sorted(output.items(), key=lambda x: (-x[1], x[0]))
                for ngram, freq in sorted_ngrams:
                    f.write("%s: %d\n" % (ngram, freq))
        except IOError:
            pass