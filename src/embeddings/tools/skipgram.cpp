#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "meta/logging/logger.h"
#include "cpptoml.h"

using namespace meta;

#define STRING_LIMIT 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000

const int vocabularyHashSize = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocabularyElement {
    long long wordFrequency;
    int *point;
    char *word, *code;
    float rand_frac;
};

char train_file[STRING_LIMIT], output_file[STRING_LIMIT];
char save_vocab_file[STRING_LIMIT], read_vocab_file[STRING_LIMIT];

struct vocabularyElement *vocabulary;
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *vocabularyHash;
long long  allocatedForVocabulary= 10000, vocabularySize = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
float alpha = 0.025, starting_alpha, sample = 0;
float *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int negative = 0;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  float d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  for (a = 0; a < vocabularySize; a++) train_words_pow += pow(vocabulary[a].wordFrequency, power);
  i = 0;
  d1 = pow(vocabulary[i].wordFrequency, power) / (float)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (float)table_size > d1) {
      i++;
      d1 += pow(vocabulary[i].wordFrequency, power) / (float)train_words_pow;
    }
    if (i >= vocabularySize) i = vocabularySize - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0;
  char ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= STRING_LIMIT - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocabularyHashSize;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocabularyHash[hash] == -1) return -1;
    if (!strcmp(word, vocabulary[vocabularyHash[hash]].word)) return vocabularyHash[hash];
    hash = (hash + 1) % vocabularyHashSize;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[STRING_LIMIT];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Expands vocabulary by new word
int ExpandVocabulary(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > STRING_LIMIT) length = STRING_LIMIT;
  vocabulary[vocabularySize].word = (char *)calloc(length, sizeof(char));
  strcpy(vocabulary[vocabularySize].word, word);
  vocabulary[vocabularySize].wordFrequency = 0;
  vocabularySize++;
  // Reallocate memory if needed
  if (vocabularySize + 2 >= allocatedForVocabulary) {
    allocatedForVocabulary += 10000;
    vocabulary = (struct vocabularyElement *)realloc(vocabulary, allocatedForVocabulary * sizeof(struct vocabularyElement));
  }
  hash = GetWordHash(word);
  while (vocabularyHash[hash] != -1) hash = (hash + 1) % vocabularyHashSize;
  vocabularyHash[hash] = vocabularySize - 1;
  return vocabularySize - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocabularyElement *)b)->wordFrequency - ((struct vocabularyElement *)a)->wordFrequency;
}

void DestroyVocab() {
  int a;

  for (a = 0; a < vocabularySize; a++) {
    if (vocabulary[a].word != NULL) {
      free(vocabulary[a].word);
    }
    if (vocabulary[a].code != NULL) {
      free(vocabulary[a].code);
    }
    if (vocabulary[a].point != NULL) {
      free(vocabulary[a].point);
    }
  }
  free(vocabulary[vocabularySize].word);
  free(vocabulary);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocabulary[1], vocabularySize - 1, sizeof(struct vocabularyElement), VocabCompare);
  for (a = 0; a < vocabularyHashSize; a++) vocabularyHash[a] = -1;
  size = vocabularySize;
  train_words = 0;
  for (a = 1; a < size; a++) { // Skip </s>
    // Words occuring less than min_count times will be discarded from the vocabulary
    if (vocabulary[a].wordFrequency < min_count) {
      vocabularySize--;
      free(vocabulary[a].word);
      vocabulary[a].word = NULL;
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocabulary[a].word);
      while (vocabularyHash[hash] != -1) hash = (hash + 1) % vocabularyHashSize;
      vocabularyHash[hash] = a;
      train_words += vocabulary[a].wordFrequency;
    }
  }
  vocabulary = (struct vocabularyElement *)realloc(vocabulary, (vocabularySize + 1) * sizeof(struct vocabularyElement));

}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocabularySize; a++) if (vocabulary[a].wordFrequency > min_reduce) {
      vocabulary[b].wordFrequency = vocabulary[a].wordFrequency;
      vocabulary[b].word = vocabulary[a].word;
      b++;
    } else free(vocabulary[a].word);
  vocabularySize = b;
  for (a = 0; a < vocabularyHashSize; a++) vocabularyHash[a] = -1;
  for (a = 0; a < vocabularySize; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocabulary[a].word);
    while (vocabularyHash[hash] != -1) hash = (hash + 1) % vocabularyHashSize;
    vocabularyHash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}


void LearnVocabFromTrainFile() {
  char word[STRING_LIMIT];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocabularyHashSize; a++) vocabularyHash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocabularySize = 0;
  ExpandVocabulary((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = ExpandVocabulary(word);
      vocabulary[a].wordFrequency = 1;
    } else vocabulary[i].wordFrequency++;
    if (vocabularySize > vocabularyHashSize * 0.7) ReduceVocab();
  }
  SortVocab();

  for (a = 1; a < vocabularySize; a++) {
    vocabulary[a].rand_frac = (sqrt(vocabulary[a].wordFrequency / (sample * train_words)) + 1) * (sample * train_words) / vocabulary[a].wordFrequency;
  }
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocabularySize);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocabularySize; i++) fprintf(fo, "%s %lld\n", vocabulary[i].word, vocabulary[i].wordFrequency);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[STRING_LIMIT];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocabularyHashSize; a++) vocabularyHash[a] = -1;
  vocabularySize = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = ExpandVocabulary(word);
    fscanf(fin, "%lld%c", &vocabulary[a].wordFrequency, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocabularySize);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  a = posix_memalign((void **)&syn0, 128, (long long)vocabularySize * layer1_size * sizeof(float));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

  a = posix_memalign((void **)&syn1neg, 128, (long long)vocabularySize * layer1_size * sizeof(float));
  if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (b = 0; b < layer1_size; b++) for (a = 0; a < vocabularySize; a++)
      syn1neg[a * layer1_size + b] = 0;

  for (b = 0; b < layer1_size; b++) for (a = 0; a < vocabularySize; a++)
      syn0[a * layer1_size + b] = (rand() / (float)RAND_MAX - 0.5) / layer1_size;

}

void DestroyNet() {
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
}


void TrainModel() {
  long a, b, c, d;
  FILE *fo;

  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();

  {
    long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)0;
    float f, g;
    clock_t now;
    float *neu1e = (float *)calloc(layer1_size, sizeof(float));
    FILE *fi = fopen(train_file, "rb");
    if (fi == NULL) {
      fprintf(stderr, "no such file or directory: %s", train_file);
      exit(1);
    }
    fseek(fi, file_size / (long long)num_threads * (long long)0, SEEK_SET);
    while (1) {
      if (word_count - last_word_count > 10000) {
        word_count_actual += word_count - last_word_count;
        last_word_count = word_count;
        if ((debug_mode > 1)) {
          now=clock();
          printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                 word_count_actual / (float)(train_words + 1) * 100,
                 word_count_actual / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000));
          fflush(stdout);
        }
        alpha = starting_alpha * (1 - word_count_actual / (float)(train_words + 1));
        if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
      }
      if (sentence_length == 0) {
        while (1) {
          word = ReadWordIndex(fi);
          if (feof(fi)) break;
          if (word == -1) continue;
          word_count++;
          if (word == 0) break;
          // The subsampling randomly discards frequent words while keeping the ranking same
          if (sample > 0) {
            //float ran = (sqrt(vocabulary[word].wordFrequency / (sample * train_words)) + 1) * (sample * train_words) / vocabulary[word].wordFrequency;
            next_random = next_random * (unsigned long long)25214903917 + 11;
            if (vocabulary[word].rand_frac < (next_random & 0xFFFF) / (float)65536) continue;
          }
          sen[sentence_length] = word;
          sentence_length++;
          if (sentence_length >= MAX_SENTENCE_LENGTH) break;
        }
        sentence_position = 0;
      }
      if (feof(fi)) break;
      if (word_count > train_words / num_threads) break;
      word = sen[sentence_position];
      if (word == -1) continue;
      for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
      next_random = next_random * (unsigned long long)25214903917 + 11;
      //b = next_random % window;
      b=0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          l1 = last_word * layer1_size;
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
          // NEGATIVE SAMPLING
          if (negative > 0) for (d = 0; d < negative + 1; d++) {
              if (d == 0) {
                target = word;
                label = 1;
              } else {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0) target = next_random % (vocabularySize - 1) + 1;
                if (target == word) continue;
                label = 0;
              }
              l2 = target * layer1_size;
              f = 0;
              for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
              if (f > MAX_EXP) g = (label - 1) * alpha;
              else if (f < -MAX_EXP) g = (label - 0) * alpha;
              else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
              for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
              for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
            }
          // Learn weights input -> hidden
          for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }
      sentence_position++;
      if (sentence_position >= sentence_length) {
        sentence_length = 0;
        continue;
      }
    }
    fclose(fi);
    free(neu1e);
  }


  fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
    exit(1);
  }
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocabularySize, layer1_size);
    for (a = 0; a < vocabularySize; a++) {
      if (vocabulary[a].word != NULL) {
        fprintf(fo, "%s ", vocabulary[a].word);
      }
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(float), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    if (centcn == NULL) {
      fprintf(stderr, "cannot allocate memory for centcn\n");
      exit(1);
    }
    int *cl = (int *)calloc(vocabularySize, sizeof(int));
    float closev, x;
    float *cent = (float *)calloc(classes * layer1_size, sizeof(float));
    for (a = 0; a < vocabularySize; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocabularySize; c++) {
        for (d = 0; d < layer1_size; d++) {
          cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
          centcn[cl[c]]++;
        }
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocabularySize; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocabularySize; a++) fprintf(fo, "%s %d\n", vocabulary[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
  free(table);
  DestroyVocab();
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;

  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;

  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " config.toml" << std::endl;
    return 1;
  }

  logging::set_cerr_logging();

//    auto config = cpptoml::parse_file(argv[1]);
  auto config = cpptoml::parse_file("config.toml");
  auto embed_cfg = config->get_table("embeddings");
  if (!embed_cfg)
  {
    std::cerr << "Missing [embeddings] configuration in " << argv[1]
              << std::endl;
    return 1;
  }

  auto skip_cfg = embed_cfg->get_table("skip-gram");

  layer1_size = skip_cfg->get_as<int64_t>("layer1_size").value_or(100); //
  debug_mode = skip_cfg->get_as<int64_t>("debug_mode").value_or(2);
  binary = skip_cfg->get_as<int64_t>("binary").value_or(0); //
  strcpy(train_file, skip_cfg->get_as<std::string>("train_file").value_or("").c_str()); //
  strcpy(save_vocab_file, skip_cfg->get_as<std::string>("save_vocab_file").value_or("").c_str());
  strcpy(read_vocab_file, skip_cfg->get_as<std::string>("read_vocab_file").value_or("").c_str());
  strcpy(output_file, skip_cfg->get_as<std::string>("output_file").value_or("").c_str()); //
  alpha = skip_cfg->get_as<double>("alpha").value_or(0.025);
  window = skip_cfg->get_as<int64_t>("window").value_or(5);
  min_count = skip_cfg->get_as<int64_t>("min_count").value_or(5);
  negative = skip_cfg->get_as<int64_t>("negative").value_or(5);
  num_threads = skip_cfg->get_as<int64_t>("num_threads").value_or(1);
  min_count = skip_cfg->get_as<int64_t>("min_count").value_or(5);
  classes = skip_cfg->get_as<int64_t>("classes").value_or(0);
  sample = skip_cfg->get_as<double>("sample").value_or(0);


  if  (negative < 2) {
    negative = 2;
  }

  if  (negative > 20) {
    negative = 20;
  }

  vocabulary = (struct vocabularyElement *)calloc(allocatedForVocabulary, sizeof(struct vocabularyElement));
  vocabularyHash = (int *)calloc(vocabularyHashSize, sizeof(int));
  expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
  if (expTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    //plot y = exp((x*2/1000 -1)*6)/(exp((x*2/1000 -1)*6) + 1)
  }

  TrainModel();
  DestroyNet();
  free(vocabularyHash);
  free(expTable);
  return 0;
}