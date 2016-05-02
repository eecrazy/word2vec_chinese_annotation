//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// 训练word2vec可同时采用层次化softmax和负采样算法两种方法来训练
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100 //一个单词最长100个字符
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
//一个句子最长1000个单词。text8中只有一行，所以连续的1000词当做一个句子来训练
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

// 词表结构
struct vocab_word {
  long long cn;//词频
  int *point;// 记录哈夫曼路径内部节点下标
  char *word, *code, codelen;//单词，哈夫曼编码，编码长度
};
// 定义训练数据，输出词向量的文件，词汇表数据结构以及输入输出文件
char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

struct vocab_word *vocab;//词汇表
int *vocab_hash;//先算出词汇的hash，vocab_hash[hash]就是word在vocab中的下标

// 定义参数的默认值
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;

real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;


// 参数定义完毕

// 用于负采样算法训练，将词汇表中的所有单词按照频次多少依次填满table表（填充内容为下标）
// 主要就是填充table表：[0,111111111111,2222222222,33333333,444444......]
// 模拟按照词频对词语进行采样
// 可以将table表打乱一下，是否效果会更好？
void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;//归一化项
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++)
    train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// 从fin文件流中读取一个单词
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  // 不到文件结尾
  while (!feof(fin))
  {
    ch = fgetc(fin);
    // 回车键
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
    {
      if (a > 0) {
        // 用换行符表示一个句子的开始符号，所以读入换行符后，还需要重新写回
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        // 第一个字符就读到了换行，说明是一个新开始的句子
        strcpy(word, (char *)"</s>");
        return;
      }
      else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  // 最后一个字符为0，标志结束符
  word[a] = 0;
}

// Returns hash value of a word，字符串hash算法
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}
// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    // 如果读入句子开始符号</s>,则返回0
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}
// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    // 重新分配内存，扩展原来的内存大小！！！
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  // 建立hash
  // 获取该词的字符串hash
  hash = GetWordHash(word);
  // 在vocab_hash里面找到一个值为-1的位置
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  // 将该位置的值设置为词表中的下标
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// 根据词频从大到小排序词汇表
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);//只释放了word的空间，cn没有释放
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }

  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
// 这一步无法指定，程序自动完成
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    }
    else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short unique binary codes
// 按照词频来构建哈夫曼树，频繁的词汇具有较短的二进制编码
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];

  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));//保存词频
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;

  // Following algorithm constructs the Huffman tree by adding one node at a time
  // 下面的算法构建哈夫曼树
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    }
    else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;//右边节点为1，左节点为2
  }

  // 下面为每一个word赋予一个哈夫曼编码
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    // 下面这段代码计算point是什么意思？
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }

  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
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
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  // 读取完成，排序词汇表
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

// 将词汇表保存到文件中
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

// 从一个保存的词汇表文件中读取词汇表，如果从训练数据构建，则不调用此函数
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();

  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  // fseek将fin指向一个新的位置，origin+offset
  // int fseek ( FILE * stream, long int offset, int origin );
  fseek(fin, 0, SEEK_END);
  // Returns the current value of the position indicator of the stream.
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  // 分配词向量空间
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

  // 随机初始化syn0，即为词向量
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
  // 层次化softmax算法
  if (hs) {
    // syn1表示哈夫曼树的内部节点向量表示
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    // 初始化syn1为全0
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
        syn1[a * layer1_size + b] = 0;
  }
  // 负采样算法
  if (negative > 0) {
    // syn1neg表示哈夫曼树的内部节点向量表示
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    // 初始化syn1neg为全0
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0;
  }
  // 负采样算法也需要构建哈夫曼树吗？
  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;

  // 这两个数据结构干嘛的？
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));

  FILE *fi = fopen(train_file, "rb");
  // 为每个线程分割训练数据，设定文件头初始位置
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);

  while (1) {

    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;

      // 输出debug信息
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               word_count_actual / (real)(iter * train_words + 1) * 100,
               word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      // 学习率逐渐减小
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      // 防止学习率过小，不小于初始学习率的万分之一
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }

    //读入训练语料中的一个句子：或者读入文件中的一行，或者读入某行中连续的1000词，下次接着读入
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;//读到句子开始符号
        // The subsampling randomly discards frequent words while keeping the ranking same
        //
        if (sample > 0) {//以一定的概率丢弃词语
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }

        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    // 迭代到达文件末尾（最后一个线程才会发生），或者本线程对应的数据已经完整迭代一次
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;//如果本线程已经迭代够iter次，则退出
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      // 重新定位到本线程对应的数据头位置，开始下一次迭代
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    word = sen[sentence_position];//一个句子中第sentence_position（0）个词在词表中的下标
    if (word == -1) continue;
    // 初始化neu1和neu1e，用于hs和negative sampling
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;//只用于cbow
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;//同时用于cbow和hs
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;//b是将window随机缩小为b

    // 开始执行训练算法
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
          cw++;
        }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        // cbow中使用层次化softmax算法
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
            f = 0;
            l2 = vocab[word].point[d] * layer1_size;
            // Propagate hidden -> output
            for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
            if (f <= -MAX_EXP) continue;
            else if (f >= MAX_EXP) continue;
            else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            // 'g' is the gradient multiplied by the learning rate
            g = (1 - vocab[word].code[d] - f) * alpha;
            // Propagate errors output -> hidden
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
            // Learn weights hidden -> output
            for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
          }

        // cbow中使用NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
            if (d == 0) {
              target = word;
              label = 1;
            } else {
              next_random = next_random * (unsigned long long)25214903917 + 11;
              target = table[(next_random >> 16) % table_size];
              if (target == 0) target = next_random % (vocab_size - 1) + 1;
              if (target == word) continue;
              label = 0;
            }
            l2 = target * layer1_size;
            f = 0;
            for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
            if (f > MAX_EXP)
              g = (label - 1) * alpha;
            else if (f < -MAX_EXP)
              g = (label - 0) * alpha;
            else
              g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
            for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
          }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
            c = sentence_position - window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;
            for (c = 0; c < layer1_size; c++)
              syn0[c + last_word * layer1_size] += neu1e[c];
          }
      }
    }//结束cbow
    else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {//此时就是j=0的情况
          c = sentence_position - window + a;//c表示要预测的单词在句子中的下标
          if (c < 0) continue;//句首的单词前面没有window那么多单词
          if (c >= sentence_length) continue;
          last_word = sen[c];//要预测的单词的下标
          if (last_word == -1) continue;
          // 要预测的词向量在syn0数组中的偏移
          l1 = last_word * layer1_size;
          // 对于窗口内的每个要预测的词，都需要初始化neu1e为0.代表反向传播的错误信息
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

          // skip-gram中使用HIERARCHICAL SOFTMAX
          if (hs)
            for (d = 0; d < vocab[word].codelen; d++) {
              f = 0;//用来存储词向量内积结果
              l2 = vocab[word].point[d] * layer1_size;//l2表示内部节点的向量表示所在偏移
              // Propagate hidden -> output
              // 向量内积
              for (c = 0; c < layer1_size; c++)
                f += syn0[c + l1] * syn1[c + l2];

              if (f <= -MAX_EXP) continue;
              else if (f >= MAX_EXP) continue;
              else
                f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
              // 'g' is the gradient multiplied by the learning rate
              g = (1 - vocab[word].code[d] - f) * alpha;

              // Propagate errors output -> hidden
              for (c = 0; c < layer1_size; c++)
                neu1e[c] += g * syn1[c + l2];
              // Learn weights hidden -> output
              for (c = 0; c < layer1_size; c++)
                syn1[c + l2] += g * syn0[c + l1];
            }
          // skip-gram中使用NEGATIVE SAMPLING
          if (negative > 0)
            for (d = 0; d < negative + 1; d++) {
              if (d == 0) {
                target = word;
                label = 1;
              } else {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0) target = next_random % (vocab_size - 1) + 1;
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
    }//结束skip-gram

    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }

  }//结束while
  //关闭文件，释放内存
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;

  // 定义多线程数据结构
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  // 初始alpha
  starting_alpha = alpha;
  // 构建词汇表
  if (read_vocab_file[0] != 0) ReadVocab();//从词汇表文件中读取词汇表
  else LearnVocabFromTrainFile();//从训练数据中构建词汇表
  // 保存词汇表
  if (save_vocab_file[0] != 0) SaveVocab();
  // 没有词向量输出文件，则退出
  if (output_file[0] == 0) return;


  // 初始化网络，分配内存
  InitNet();

  // 用负采样来训练，构建unigram表，用于以一定概率采样负样本
  if (negative > 0) InitUnigramTable();
  // 记录开始时间
  start = clock();

  // 开启多线程进行训练
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

  // 输出词向量的文件
  fo = fopen(output_file, "wb");

  // 训练结束，输出词向量
  if (classes == 0) {//无需聚类
    // Save the word vectors
    // 输出词汇表大小和词向量的维度
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      // 二进制形式输出
      if (binary)
        // 输出二进制形式的词向量，这句二进制输出的代码含义是什么？
        for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      // 文本格式输出
      else
        for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  }
  // 输出聚类结果，共classes个类别
  else {
    // Run K-means on the word vectors，聚成classes个类
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));

    // 每个词随机初始化一个类别
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;

    // K-means聚类算法
    for (a = 0; a < iter; a++)
    {
      for (b = 0; b < clcn * layer1_size; b++)
        cent[b] = 0;
      for (b = 0; b < clcn; b++)
        centcn[b] = 1;
      for (c = 0; c < vocab_size; c++)
      {
        for (d = 0; d < layer1_size; d++)
          cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++)
      {
        closev = 0;
        for (c = 0; c < layer1_size; c++)
        {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++)
      {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++)
        {
          x = 0;
          for (b = 0; b < layer1_size; b++)
            x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev)
          {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}
// 解析参数的下标
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        //找到了参数名字，却没有提供数值
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      //返回字符串下标
      return a;
    }
  // 没有提供参数
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  // 初始化词向量，词汇表输入输出文件名为空
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;

  // 解析参数
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

  // 为数据结构分配空间
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));


  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  // 预先计算expTable,表示sigma函数在x<[-6,6]时的函数值
  for (i = 0; i <= EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table，[exp(-6),exp(6)]
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1),[sigma(-6),sigma(6)]
  }
  TrainModel();
  return 0;
}


