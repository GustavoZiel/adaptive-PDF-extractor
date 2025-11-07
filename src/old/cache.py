from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

from logger import get_logger
from rule import Rule

logger = get_logger(__name__)


class CacheItem:
    def __init__(self, rule: Rule, weight: int = 1):
        self.rule = rule  # Now stores a Rule object, not a string
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
        """Apply the Rule object to text and return the extracted value."""
        return self.rule.apply(text)

    def validate(self, text: str) -> bool:
        """Validate the extracted text using the Rule object's validation."""
        return self.rule.validate(text)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "rule": self.rule.model_dump(),  # Serialize Rule as dict
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

    def add_rule(self, rule: Rule, weight: int = 1):
        """Add a new Rule object with weight."""
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
            cache_item = node.item
            extracted_text = cache_item.apply(text)
            if cache_item.validate(extracted_text):
                logger.debug(
                    "Rule matched - Type: %s, Current weight: %d",
                    cache_item.rule.type,
                    cache_item.weight,
                )
                cache_item.increment()
                logger.debug(
                    "✓ Incremented rule weight: %d → %d",
                    cache_item.weight - 1,
                    cache_item.weight,
                )
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

    def __len__(self):
        return self.length

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
        rules = self.fields[field]
        extracted_text = rules.try_extract(text)
        return extracted_text

    def save_to_file_json(self, filename: str, filepath: str):
        """Save cache to a JSON file."""
        data = {
            field: rules_list.get_data() for field, rules_list in self.fields.items()
        }
        os.makedirs(os.path.dirname(os.path.join(filepath, filename)), exist_ok=True)
        output_path = os.path.join(filepath, filename)
        with open(output_path, "w", encoding="utf-8") as f:
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
                # Deserialize Rule object from dict
                rule_obj = Rule.model_validate(item["rule"])
                rules_list.add_rule(rule=rule_obj, weight=item["weight"])
            instance.fields[field] = rules_list
        return instance

    def save_to_file_pickle(self, filename: str, filepath: str):
        """Save cache to a pickle file."""
        data = {
            field: rules_list.get_data() for field, rules_list in self.fields.items()
        }
        os.makedirs(os.path.dirname(os.path.join(filepath, filename)), exist_ok=True)
        output_path = os.path.join(filepath, filename)
        with open(output_path, "wb") as f:
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
                # Deserialize Rule object from dict
                rule_obj = Rule.model_validate(item["rule"])
                rules_list.add_rule(rule=rule_obj, weight=item["weight"])
            instance.fields[field] = rules_list
        return instance

    def __repr__(self):
        return f"Cache(fields={list(self.fields.keys())})"
