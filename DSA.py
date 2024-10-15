import torch                       
import torch.nn as nn             
import torch.nn.functional as F     
from torch.utils.data import Dataset, DataLoader 
import numpy as np                
import random 
import torch.optim as optim
import string



# Transformer model block

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x):
        attn_output, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.dropout(attn_output)  
        x = x + self.dropout(self.mlp(self.ln2(x)))  
        
        return x

# tokenizer .the intial tokenizer was just for the chracter level but it will not be enough for me


class AdvancedTokenizer:
    def __init__(self, lowercase=True):
        chars = string.ascii_letters + string.digits + string.punctuation + " "
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.pad_token = len(self.stoi) 
        self.stoi['[PAD]'] = self.pad_token
        self.itos[self.pad_token] = '[PAD]'
        
        self.unk_token = self.pad_token + 1  
        self.stoi['[UNK]'] = self.unk_token
        self.itos[self.unk_token] = '[UNK]'
        
        self.cls_token = self.pad_token + 2  
        self.stoi['[CLS]'] = self.cls_token
        self.itos[self.cls_token] = '[CLS]'
        
        self.sep_token = self.pad_token + 3  
        self.stoi['[SEP]'] = self.sep_token
        self.itos[self.sep_token] = '[SEP]'
        
        self.vocab_size = len(self.stoi)
        self.lowercase = lowercase  

    def encode(self, text, max_len=None):
        if self.lowercase:
            text = text.lower()

        tokens = [self.stoi.get(c, self.unk_token) for c in text]
        tokens = [self.cls_token] + tokens + [self.sep_token]  

        if max_len:
            tokens = tokens[:max_len]  
            tokens += [self.pad_token] * (max_len - len(tokens))  
        return tokens

    def decode(self, token_ids):
        tokens = [self.itos.get(id, '[UNK]') for id in token_ids]
        return ''.join(tokens).replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '')

    def batch_encode(self, texts, max_len=None):
        if max_len is None:
            max_len = self.max_seq_len  
        return [self.encode(text, max_len=max_len) for text in texts]


    def batch_decode(self, batch_token_ids):
        return [self.decode(token_ids) for token_ids in batch_token_ids]
    
# solution generator nn


class DSASolutionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_seq_len, dropout=0.1):
        super(DSASolutionGenerator, self).__init__()
        self.embed_size = embed_size
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_seq_len, embed_size)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        token_embeds = self.token_embedding(x)
        position_ids = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        position_embeds = self.position_embedding(position_ids)
        
        x = self.dropout(token_embeds + position_embeds)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        output = self.fc_out(x)
        return output




  
# dataset block 


class DSADataset(Dataset):
    def __init__(self, problems, solutions, tokenizer, max_seq_len):
        self.problems = problems
        self.solutions = solutions
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]
        solution = self.solutions[idx]
        problem_ids = self.tokenizer.encode(problem, max_len=self.max_seq_len)
        solution_ids = self.tokenizer.encode(solution, max_len=self.max_seq_len)
        problem_tensor = torch.tensor(problem_ids, dtype=torch.long)
        solution_tensor = torch.tensor(solution_ids, dtype=torch.long)

        return problem_tensor, solution_tensor

# for training one question at a time

def train_single_example(model, tokenizer, problem, solution, optimizer, loss_fn, max_seq_len):
    model.train()
    
    # Create a temporary dataset with a single example
    temp_dataset = DSADataset([problem], [solution], tokenizer, max_seq_len)
    temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False)
    
    # Get the single example from the loader
    problem_tensor, solution_tensor = next(iter(temp_loader))
    problem_tensor, solution_tensor = problem_tensor.to(model.device), solution_tensor.to(model.device)

    optimizer.zero_grad()
    logits = model(problem_tensor)
    logits = logits.view(-1, logits.size(-1))
    solution_tensor = solution_tensor.view(-1)

    loss = loss_fn(logits, solution_tensor)
    loss.backward()
    optimizer.step()

    return loss.item()




# Training block

def train(model, train_dataset, val_dataset, tokenizer, epochs, batch_size, learning_rate, max_seq_len):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token)

    for epoch in range(epochs):
        model.train()  
        train_loss = 0

        for problem, solution in train_loader:
            problem, solution = problem.to(model.device), solution.to(model.device)

            optimizer.zero_grad() 
            logits = model(problem)
            logits = logits.view(-1, logits.size(-1))  
            solution = solution.view(-1) 

            loss = loss_fn(logits, solution)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()  
        val_loss = 0
        with torch.no_grad():
            for problem, solution in val_loader:
                problem, solution = problem.to(model.device), solution.to(model.device)

                logits = model(problem)
                logits = logits.view(-1, logits.size(-1))
                solution = solution.view(-1)

                loss = loss_fn(logits, solution)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    print("Training completed successfully!")

# generating block

def generate_solution(model, tokenizer, problem, max_tokens=100):
    model.eval()
    input_ids = tokenizer.encode(problem, max_len=model.max_seq_len)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(model.device)

    generated_ids = input_tensor.clone()
    for _ in range(max_tokens):
        if generated_ids.size(1) >= model.max_seq_len:
            break
        
        logits = model(generated_ids[:, -model.max_seq_len:])  # Only use the last max_seq_len tokens
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()

        generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(model.device)], dim=1)
        if next_token_id == tokenizer.pad_token:
            break
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text



# training the model

tokenizer = AdvancedTokenizer()
model = DSASolutionGenerator(vocab_size=tokenizer.vocab_size, embed_size=256, num_heads=8, num_layers=4, max_seq_len=50)

problems = [
    "Implement a binary search tree",
    "Write a function to find the nth Fibonacci number",
    "Implement a quicksort algorithm",
    "Write a function to reverse a linked list",
    "Implement a stack using arrays",
    "Write a function to find the maximum element in a binary tree",
    "Implement a breadth-first search algorithm",
    "Write a function to check if a string is a palindrome",
    "Implement a hash table",
    "Write a function to find the longest common subsequence of two strings"
]

solutions = [
    """
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    """,
    """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
    """,
    """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
    """,
    """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
    """,
    """
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def peek(self):
        if not self.is_empty():
            return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
    """,
    """
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def find_max_element(root):
    if root is None:
        return float('-inf')
    
    left_max = find_max_element(root.left)
    right_max = find_max_element(root.right)
    
    return max(root.val, left_max, right_max)
    """,
    """
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    """,
    """
def is_palindrome(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]
    """,
    """
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(self.size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        hash_index = self._hash(key)
        for item in self.table[hash_index]:
            if item[0] == key:
                item[1] = value
                return
        self.table[hash_index].append([key, value])

    def get(self, key):
        hash_index = self._hash(key)
        for item in self.table[hash_index]:
            if item[0] == key:
                return item[1]
        raise KeyError(key)

    def remove(self, key):
        hash_index = self._hash(key)
        for i, item in enumerate(self.table[hash_index]):
            if item[0] == key:
                del self.table[hash_index][i]
                return
        raise KeyError(key)
    """,
    """
def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]"""]

train_dataset = DSADataset(problems, solutions, tokenizer, max_seq_len=50)


train(model, train_dataset, train_dataset, tokenizer, epochs=10, batch_size=2, learning_rate=0.001, max_seq_len=50)

