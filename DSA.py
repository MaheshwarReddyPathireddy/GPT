import torch                       
import torch.nn as nn             
import torch.nn.functional as F     
from torch.utils.data import Dataset, DataLoader 
import numpy as np                
import random 
import torch.optim as optim
import string
import pickle
import os
from sklearn.model_selection import train_test_split


# utility functions
# added them to save the current state of the model and save the current dataset

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    model.to(model.device)

def save_dataset(dataset, path):
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)



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
    def __init__(self, lowercase=True, max_seq_len=50):
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
        self.max_seq_len = max_seq_len  

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


def train(model, train_dataset, val_dataset, tokenizer, epochs, batch_size, learning_rate, max_seq_len, model_path='model.pth', dataset_path='dataset.pkl'):
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

    save_model(model, model_path)
    save_dataset((problems, solutions), dataset_path)
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

tokenizer = AdvancedTokenizer(max_seq_len=50)
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
    "Write a function to find the longest common subsequence of two strings",
    "Reverse the Array: Given an array, reverse the order of its elements in place.",
    "Maximum-Subarray: Find the contiguous subarray within an array (containing at least one number) which has the largest sum.",
    "Contains Duplicate: Given an array of integers, find if the array contains any duplicates.",
    "Chocolate Distribution Problem: Given an array of integers representing the number of chocolates in each packet, distribute the chocolates among students such that the difference between the maximum and minimum number of chocolates given to any student is minimized.",
    "Search in Rotated Sorted Array: Given a sorted array that has been rotated at an unknown pivot index, find if a target value exists in the array.",
    "Next Permutation: Implement the next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.",
    "Best time to Buy and Sell Stock: Given an array of stock prices, find the maximum profit that can be achieved by buying and selling the stock once.",
    "Repeat and Missing Number Array: Given an array of size N, where each element is in the range [1, N], find the repeating and missing numbers.",
    "Kth-Largest Element in an Array: Find the kth largest element in an unsorted array.",
    "Trapping Rain Water: Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.",
    "Product of Array Except Self: Given an array nums, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].",
    "Maximum Product Subarray: Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.",
    "Find Minimum in Rotated Sorted Array: Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. Find the minimum element.",
    "Find Pair with Sum in Sorted & Rotated Array: Given a sorted and rotated array, find a pair with a given sum.",
    "3Sum: Given an array nums of n integers, find all unique triplets in the array which gives the sum of zero.",
    "Container With Most Water: Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0). Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.",
    "Given Sum Pair: Given an array of integers and a target sum, find a pair of numbers in the array that add up to the target sum.",
    "Kth - Smallest Element: Find the kth smallest element in an unsorted array.",
    "Merge Overlapping Intervals: Given a collection of intervals, merge all overlapping intervals.",
    "Find Minimum Number of Merge Operations to Make an Array Palindrome: Given an array of positive integers, find the minimum number of merge operations required to make the array a palindrome.",
    "Given an Array of Numbers Arrange the Numbers to Form the Biggest Number: Given an array of non-negative integers, arrange them such that they form the largest number.",
    "Space Optimization Using Bit Manipulations: Given an array of integers, use bit manipulations to optimize space usage.",
    "Subarray Sum Divisible K: Given an array of integers and a number k, find the total number of continuous subarrays whose sum is divisible by k.",
    "Print all Possible Combinations of r Elements in a Given Array of Size n: Given an array of size n, generate all possible combinations of r elements.",
    "Mo's Algorithm: Given an array of integers and a set of queries, use Mo's Algorithm to efficiently answer the queries.",
    " Given a string s, return true if it is a palindrome, or false otherwise"
    
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

    return dp[m][n]
    """,
    """
def reverse_array(arr):
    return arr[::-1]
    """,
    """
def max_subarray(nums):
    max_ending_here = max_so_far = nums[0]
    for x in nums[1:]:
        max_ending_here = max(x, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far
    """,
    """
def contains_duplicate(nums):
    return len(nums) != len(set(nums))
    """,
    """
def chocolate_distribution(arr, m):
    arr.sort()
    min_diff = float('inf')
    for i in range(len(arr) - m + 1):
        min_diff = min(min_diff, arr[i + m - 1] - arr[i])
    return min_diff
    """,
    """
def search_rotated_sorted_array(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
    """,
    """
def next_permutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1:] = reversed(nums[i + 1:])
    return nums
    """,
    """
def max_profit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        profit = price - min_price
        max_profit = max(max_profit, profit)
    return max_profit
    """,
    """
def find_repeat_missing(nums):
    n = len(nums)
    xor = 0
    for i in range(1, n + 1):
        xor ^= i
    for num in nums:
        xor ^= num
    set_bit = xor & ~(xor - 1)
    x = 0
    y = 0
    for i in range(1, n + 1):
        if i & set_bit:
            x ^= i
        else:
            y ^= i
    for num in nums:
        if num & set_bit:
            x ^= num
        else:
            y ^= num
    return x, y
    """,
    """
import heapq
def kth_largest_element(nums, k):
    return heapq.nlargest(k, nums)[-1]
    """,
    """
def trap_rain_water(height):
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            water += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water += right_max - height[right]
    return water
    """,
    """
def product_except_self(nums):
    n = len(nums)
    output = [1] * n
    left = 1
    for i in range(n):
        output[i] *= left
        left *= nums[i]
    right = 1
    for i in range(n - 1, -1, -1):
        output[i] *= right
        right *= nums[i]
    return output
    """,
    """
def max_product_subarray(nums):
    max_product = min_product = result = nums[0]
    for i in range(1, len(nums)):
        if nums[i] < 0:
            max_product, min_product = min_product, max_product
        max_product = max(nums[i], max_product * nums[i])
        min_product = min(nums[i], min_product * nums[i])
        result = max(result, max_product)
    return result
    """,
    """
def find_min_rotated_sorted_array(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]
    """,
    """
def find_pair_with_sum(arr, target):
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] + arr[j] == target:
                return (arr[i], arr[j])
    return None
    """,
    """
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return result
    """,
    """
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        max_water = max(max_water, min(height[left], height[right]) * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water
    """,
    """
def given_sum_pair(arr, target):
    seen = set()
    for num in arr:
        complement = target - num
        if complement in seen:
            return (complement, num)
        seen.add(num)
    return None
    """,
    """
import heapq
def kth_smallest_element(nums, k):
    return heapq.nsmallest(k, nums)[-1]
    """,
    """
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    return merged
    """,
    """
def min_merge_operations(arr):
    i, j = 0, len(arr) - 1
    merge_ops = 0
    while i <= j:
        if arr[i] == arr[j]:
            i += 1
            j -= 1
        elif arr[i] < arr[j]:
            i += 1
            arr[i] += arr[i - 1]
            merge_ops += 1
        else:
            j -= 1
            arr[j] += arr[j + 1]
            merge_ops += 1
    return merge_ops
    """,
    """
from functools import cmp_to_key
def largest_number(nums):
    nums = list(map(str, nums))
    nums.sort(key=cmp_to_key(lambda x, y: ((y + x) > (x + y)) - ((y + x) < (x + y))))
    return ''.join(nums).lstrip('0') or '0'
    """,
    """
def space_optimization_bit_manipulations(arr):
    xor = 0
    for num in arr:
        xor ^= num
    return xor
    """,
    """
def subarray_sum_divisible_k(nums, k):
    count = 0
    prefix_sum = 0
    mod_count = [0] * k
    mod_count[0] = 1
    for num in nums:
        prefix_sum = (prefix_sum + num) % k
        count += mod_count[prefix_sum]
        mod_count[prefix_sum] += 1
    return count
    """,
    """
from itertools import combinations
def combinations_r_elements(arr, r):
    return list(combinations(arr, r))
    """,
    """
def mo_algorithm(arr, queries):
    results = []
    for query in queries:
        left, right = query
        results.append(sum(arr[left:right + 1]))
    return results
    """,
    """
def is_palindrome(s: str) -> bool:
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]
    """
]


model_path = 'model.pth'
dataset_path = 'dataset.pkl'

tokenizer = AdvancedTokenizer(max_seq_len=50)

model = DSASolutionGenerator(vocab_size=tokenizer.vocab_size, embed_size=256, num_heads=8, num_layers=4, max_seq_len=50)

if os.path.exists(model_path):
    print("Loading existing model and dataset...")
    load_model(model, model_path)
    problems, solutions = load_dataset(dataset_path)
else:
    print("No existing model found. Training from scratch...")
    problem = []
    solutions = []
# for printing the dataset
# commented it and only using whenever required
"""print("Current Dataset:")
for pro, sol in zip(problems, solutions):
    print(f"Problem: {pro}")
    print(f"Solution: {sol}")
    print()
"""
initial_length = len(problems)

# place for adding new problems and solutions


new_problems = [
    "Variables in Python",
    "Dynamic typing in Python",
    "Variable naming conventions",
    "Data types in Python",
    "Variable scope in Python",
    "Variable operations in Python",
    "Variable deletion in Python",
    "Best practices for using variables in Python"
]

new_solutions = [
    """
    Variables in Python are created by assigning values. No type declaration is needed.
    Code:
    x = 10  # x is now an integer
    y = "Hello"  # y is now a string
    """,
    """
    Python is dynamically typed, meaning variable types are inferred from assigned values.
    Code:
    x = 10  # x is an integer
    x = "Hello"  # x is now a string
    """,
    """
    Python variables should follow naming conventions like snake_case for variables and CONSTANT_CASE for constants.
    Code:
    my_variable = 10  # snake_case
    MAX_VALUE = 100  # CONSTANT_CASE
    """,
    """
    Python supports data types like int, float, str, list, tuple, and dict.
    Code:
    int_var = 10  # integer
    float_var = 3.14  # float
    str_var = "Hello"  # string
    list_var = [1, 2, 3]  # list
    tuple_var = (1, 2, 3)  # tuple
    dict_var = {"key": "value"}  # dictionary
    """,
    """
    Variables can have local, global, or nonlocal scope depending on where they are defined.
    Code:
    global_var = 10  # global variable

    def my_function():
        local_var = 20  # local variable
        print(local_var)

    def outer_function():
        nonlocal_var = 30  # nonlocal variable
        def inner_function():
            nonlocal nonlocal_var
            nonlocal_var += 1
            print(nonlocal_var)
        inner_function()
    """,
    """
    Variables support various operations like assignment, arithmetic, comparison, logical, identity, and membership.
    Code:
    x = 10  # assignment
    x += 5  # arithmetic operation
    y = 15
    print(x == y)  # comparison operation
    print(x > 5 and y < 20)  # logical operation
    print(x is y)  # identity operation
    print(x in [10, 20, 30])  # membership operation
    """,
    """
    Variables can be deleted using the `del` statement to remove them from their scope.
    Code:
    x = 10
    del x  # x is deleted
    """,
    """
    Best practices include using descriptive names, avoiding single-letter variables, and maintaining consistent naming styles.
    Code:
    # Good practice
    user_age = 25
    total_amount = 1000

    # Avoid single-letter variables
    # Bad practice
    a = 25
    b = 1000
    """
]

if new_problems != problems[-len(new_problems):] or new_solutions != solutions[-len(new_solutions):]:
    problems.extend(new_problems)
    solutions.extend(new_solutions)
else:
    print("No new data to train the model. Add new data to continue training.")
    exit()

if len(problems) == initial_length:
    print("No new data to train the model. Add new data to continue training.")
else:
   
    problems1_train, problems1_val, solutions1_train, solutions1_val = train_test_split(problems, solutions, test_size=0.2, random_state=42)

    train_dataset = DSADataset(problems1_train, solutions1_train, tokenizer, max_seq_len=50)
    val_dataset = DSADataset(problems1_val, solutions1_val, tokenizer, max_seq_len=50)

    train(model, train_dataset, val_dataset, tokenizer, epochs=10, batch_size=2, learning_rate=0.001, max_seq_len=50, model_path=model_path, dataset_path=dataset_path)



