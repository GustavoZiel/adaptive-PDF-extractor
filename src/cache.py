from __future__ import annotations

import json
import pickle
from collections import defaultdict

from rule import Rule


class CacheItem:
    def __init__(self, rule: str, weight: int = 1):
        self.rule = rule
        self.weight = weight

    def increment(self, value: int = 1) -> "CacheItem":
        """Increment weight by value."""
        self.weight += value
        return self

    def decrement(self, value: int = 1) -> "CacheItem":
        """Decrement weight by value."""
        self.weight -= value
        return self

    def apply(self, text: str) -> str:
        """Apply rule to text, return text if matches (case-insensitive), else empty string."""
        if text.lower() == self.rule.lower():
            return text
        return ""

    def validate(self, text: str) -> bool:
        """Validate if text matches rule (case-insensitive)."""
        return text.lower() == self.rule.lower()

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "rule": str(self.rule),
            "weight": self.weight,
        }

    def __eq__(self, other: "CacheItem") -> bool:
        return self.weight == other.weight

    def __lt__(self, other: "CacheItem") -> bool:
        return self.weight < other.weight

    def __gt__(self, other: "CacheItem") -> bool:
        return self.weight > other.weight

    def __repr__(self):
        return f"{self.__class__.__name__}(rule={self.rule}, weight={self.weight})"


class Node:
    def __init__(self, item: CacheItem):
        self.item = item
        self.prev: Node | None = None
        self.next: Node | None = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.item.__repr__()})"


class RulesList:
    def __init__(self):
        self.head: Node | None = None
        self.curr: Node | None = None
        self.length: int = 0

    def add_rule(self, rule: str, weight: int = 1):
        """Add a new rule with weight."""
        node = Node(item=CacheItem(rule=rule, weight=weight))
        if not self.head:
            self.head = node
            self.curr = node
        else:
            self.curr.next = node
            node.prev = self.curr
            self.curr = node
        self.length += 1

    def try_extract(self, text: str) -> str | None:
        """Try to extract text using rules, increment weight on match."""
        for node in self:
            rule = node.item
            extracted_text = rule.apply(text)
            if rule.validate(extracted_text):
                rule.increment()
                self.update(node)
                return extracted_text
        return None

    def update(self, node: Node):
        """Update node position by bubbling up based on weight."""
        if not node.prev:
            return

        while node.prev and node.prev.item < node.item:
            prev = node.prev
            prev_prev = prev.prev
            next = node.next

            if prev_prev:
                prev_prev.next = node
            else:
                self.head = node
            node.prev = prev_prev

            if next:
                next.prev = prev
            else:
                self.curr = prev
            prev.next = next

            node.next = prev
            prev.prev = node

    def get_data(self) -> list[CacheItem]:
        """Get list of cache items as dictionaries."""
        data = []
        for aux in self:
            data.append(aux.item.to_dict())
        return data

    def __repr__(self):
        list_repr = []
        for node in self:
            list_repr.append(repr(node))
        nodes_repr = ",".join(list_repr)
        return f"{self.__class__.__name__}(nodes=[{nodes_repr}])"

    def __iter__(self):
        current = self.head
        while current:
            yield current
            current = current.next


class Cache:
    def __init__(self):
        self.fields = defaultdict(RulesList)

    def add_rule(self, field, rule):
        """Add a rule to a field."""
        self.fields[field].add_rule(rule)

    def try_extract(self, field, text):
        """Try extracting using cached rules for a field."""
        rules = self.fields.get(field)
        extracted_text = rules.try_extract(text)
        return extracted_text

    def save_to_file_json(self, filepath: str):
        """Save cache to JSON file."""
        data = {}
        for field, rules_list in self.fields.items():
            data[field] = rules_list.get_data()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file_json(cls, filepath: str):
        """Load cache from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls()
        for field, items in data.items():
            rules_list = RulesList()
            for item in items:
                rules_list.add_rule(rule=item["rule"], weight=item["weight"])
            instance.fields[field] = rules_list
        return instance

    def save_to_file_pickle(self, filepath: str):
        """Save cache to pickle file."""
        data = {}
        for field, rules_list in self.fields.items():
            data[field] = rules_list.get_data()
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_from_file_pickle(cls, filepath: str):
        """Load cache from pickle file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        for field, items in data.items():
            rules_list = RulesList()
            for item in items:
                rules_list.add_rule(rule=item["rule"], weight=item["weight"])
            instance.fields[field] = rules_list
        return instance

    def __repr__(self):
        return f"Cache(fields={list(self.fields.keys())})"
