# Develop a MapReduce program to calculate the frequency of a given word in a given file.

# python word_frequency.py input.txt > output.txt
# Assuming you have a file named input.txt containing the text you want to analyze, you can run the MapReduce job using the following command:

from mrjob.job import MRJob

class WordFrequency(MRJob):
    def mapper(self, _, line):
        # Split each line into words
        words = line.split()
        # Emit key-value pairs of (word, 1) for each word
        for word in words:
            yield word.lower(), 1

    def reducer(self, word, counts):
        # Sum up the counts for each word
        yield word, sum(counts)

if __name__ == '__main__':
    WordFrequency.run()
