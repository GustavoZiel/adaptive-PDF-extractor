"""Adaptive LRU cache system for extraction rules.

This module implements a weighted LRU (Least Recently Used) cache for storing
and prioritizing extraction rules. Rules that successfully extract values are
automatically promoted (increased weight), making the cache adaptive to the
most effective rules.

Key components:
- CacheItem: Wraps a Rule with success weight tracking
- Node: Linked list node for LRU implementation
- RulesList: Per-field LRU cache with automatic priority reordering
- Cache: Top-level container managing RulesList for each field
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

from logger import get_logger
from rule import Rule

logger = get_logger(__name__)


# ============================================================================
# SECTION 1: Cache Item (Rule Wrapper)
# ============================================================================


class CacheItem:
    """Wraps a Rule with weight tracking for LRU prioritization.

    The weight increases each time the rule successfully extracts a value,
    causing it to bubble up in the priority queue for faster future access.

    Attributes:
        rule: The Rule object containing extraction logic
        weight: Success counter (higher = more successful/prioritized)
    """

    def __init__(self, rule: Rule, weight: int = 1):
        """Initialize cache item with a rule and optional weight.

        Args:
            rule: Rule object to wrap
            weight: Initial weight/priority (default: 1)
        """
        self.rule = rule
        self.weight = weight

    def increment(self, value: int = 1) -> "CacheItem":
        """Increment weight by value (called on successful extraction).

        Args:
            value: Amount to increment weight by (default: 1)

        Returns:
            Self for chaining
        """
        self.weight += value
        return self

    def decrement(self, value: int = 1) -> "CacheItem":
        """Decrement weight by value.

        Args:
            value: Amount to decrement weight by (default: 1)

        Returns:
            Self for chaining
        """
        self.weight -= value
        return self

    def apply(self, text: str) -> str:
        """Apply the wrapped Rule to text and return extracted value.

        Args:
            text: Input text to extract from

        Returns:
            Extracted value or None if extraction fails
        """
        return self.rule.apply(text)

    def validate(self, text: str) -> bool:
        """Validate extracted text using the Rule's validation regex.

        Args:
            text: Extracted text to validate

        Returns:
            True if text matches validation pattern, False otherwise
        """
        return self.rule.validate(text)

    def to_dict(self):
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with rule (as dict) and weight
        """
        return {
            "rule": self.rule.model_dump(),  # Serialize Rule as dict
            "weight": self.weight,
        }

    def __eq__(self, other: "CacheItem") -> bool:
        """Compare by weight for equality."""
        return self.weight == other.weight

    def __lt__(self, other: "CacheItem") -> bool:
        """Compare by weight (less than)."""
        return self.weight < other.weight

    def __gt__(self, other: "CacheItem") -> bool:
        """Compare by weight (greater than)."""
        return self.weight > other.weight

    def __repr__(self):
        return f"{self.__class__.__name__}(rule={self.rule}, weight={self.weight})"


# ============================================================================
# SECTION 2: Linked List Node
# ============================================================================


class Node:
    """Doubly-linked list node for LRU cache implementation.

    Each node wraps a CacheItem and maintains pointers to previous
    and next nodes in the priority-ordered list.

    Attributes:
        item: CacheItem stored in this node
        prev: Previous node in the list (higher priority)
        next: Next node in the list (lower priority)
    """

    def __init__(self, item: CacheItem):
        """Initialize node with a cache item.

        Args:
            item: CacheItem to store in this node
        """
        self.item = item
        self.prev: Node | None = None
        self.next: Node | None = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.item.__repr__()})"


# ============================================================================
# SECTION 3: Rules List (Per-Field LRU Cache)
# ============================================================================


class RulesList:
    """LRU cache for rules of a specific field.

    Maintains a doubly-linked list of rules ordered by weight (priority).
    When a rule successfully extracts a value, its weight increases and
    it bubbles up in the list for faster future access.

    Attributes:
        head: First node in the list (highest priority)
        curr: Last node in the list (most recently added)
        length: Number of rules in the list
    """

    def __init__(self):
        """Initialize empty rules list."""
        self.head: Node | None = None
        self.curr: Node | None = None
        self.length: int = 0

    def add_rule(self, rule: Rule, weight: int = 1):
        """Add a new Rule to the end of the list.

        Args:
            rule: Rule object to add
            weight: Initial weight for the rule (default: 1)
        """
        node = Node(item=CacheItem(rule=rule, weight=weight))
        if not self.head:
            # First rule in the list
            self.head = node
            self.curr = node
        else:
            # Append to end
            self.curr.next = node
            node.prev = self.curr
            self.curr = node
        self.length += 1
        logger.debug(
            "Added rule to cache - Type: %s, Weight: %d, Total rules: %d",
            rule.type,
            weight,
            self.length,
        )

    def try_extract(self, text: str) -> str | None:
        """Try extracting text using rules in priority order.

        Iterates through rules from highest to lowest priority. On successful
        extraction, increments the rule's weight and reorders the list.

        Args:
            text: Input text to extract from

        Returns:
            Extracted value or None if no rule succeeds
        """
        for node in self:
            cache_item = node.item
            extracted_text = cache_item.apply(text)

            # Check if extraction was successful and valid
            if cache_item.validate(extracted_text):
                # logger.debug(
                #     "Rule matched - Type: %s, Current weight: %d",
                #     cache_item.rule.type,
                #     cache_item.weight,
                # )
                old_weight = cache_item.weight
                cache_item.increment()
                # logger.debug(
                #     "✓ Incremented rule weight: %d → %d", old_weight, cache_item.weight
                # )

                # Bubble up in priority list
                self.update(node)

                return extracted_text

        # No rule matched
        logger.debug("No cached rule matched for this field")
        return None

    def update(self, node: Node):
        """Bubble node up in the list based on increased weight.

        This maintains the priority ordering where higher-weight rules
        are checked first.

        Args:
            node: Node to reposition in the list
        """
        if not node.prev:
            # Already at head, no need to move
            return

        # Bubble up while node has higher weight than predecessor
        while node.prev and node.prev.item < node.item:
            prev = node.prev
            prev_prev = prev.prev
            next = node.next

            # Update head if node becomes new head
            if prev_prev:
                prev_prev.next = node
            else:
                self.head = node
            node.prev = prev_prev

            # Update tail if node was at tail
            if next:
                next.prev = prev
            else:
                self.curr = prev
            prev.next = next

            # Swap node and prev
            node.next = prev
            prev.prev = node

        logger.debug(
            "Rule reordered - New weight: %d (bubbled up in priority)",
            node.item.weight,
        )

    def get_data(self) -> list[dict]:
        """Get list of cache items as dictionaries for serialization.

        Returns:
            List of dictionaries with rule and weight information
        """
        data = []
        for node in self:
            data.append(node.item.to_dict())
        return data

    def __len__(self):
        """Return number of rules in the list."""
        return self.length

    def __repr__(self):
        list_repr = []
        for node in self:
            list_repr.append(repr(node))
        nodes_repr = ",".join(list_repr)
        return f"{self.__class__.__name__}(nodes=[{nodes_repr}])"

    def __iter__(self):
        """Iterate through nodes from head to tail (highest to lowest priority)."""
        current = self.head
        while current:
            yield current
            current = current.next


# ============================================================================
# SECTION 4: Cache (Top-Level Container)
# ============================================================================


class Cache:
    """Top-level cache container managing RulesList for each field.

    Each field has its own independent RulesList with priority-ordered rules.
    Provides convenient methods for adding rules, extracting values, and
    saving/loading the entire cache structure.

    Attributes:
        fields: Dictionary mapping field names to RulesList instances
    """

    def __init__(self):
        """Initialize empty cache with defaultdict of RulesLists."""
        self.fields = defaultdict(RulesList)

    def add_rule(self, field: str, rule: Rule):
        """Add a rule to a specific field's cache.

        Args:
            field: Field name to add rule to
            rule: Rule object to add
        """
        self.fields[field].add_rule(rule)
        logger.debug("Added rule to field '%s'", field)

    def try_extract(self, field: str, text: str) -> str | None:
        """Try extracting a field value using cached rules.

        Args:
            field: Field name to extract
            text: Input text to extract from

        Returns:
            Extracted value or None if no rule succeeds
        """
        rules_list = self.fields[field]
        extracted_text = rules_list.try_extract(text)
        return extracted_text

    # ========================================================================
    # Serialization Methods
    # ========================================================================

    def save_to_file_json(self, filename: str, filepath: str):
        """Save cache to a JSON file.

        Args:
            filename: Name of the output file
            filepath: Directory path to save to
        """
        data = {
            field: rules_list.get_data() for field, rules_list in self.fields.items()
        }

        # Ensure directory exists
        output_dir = os.path.dirname(os.path.join(filepath, filename))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(filepath, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Cache saved to JSON: %s", output_path)

    @classmethod
    def load_from_file_json(cls, filepath: str) -> "Cache":
        """Load cache from JSON file.

        Args:
            filepath: Path to the JSON file to load

        Returns:
            Cache instance populated with loaded rules
        """
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

        logger.info(
            "Cache loaded from JSON: %s (%d fields)",
            filepath,
            len(instance.fields),
        )
        return instance

    def save_to_file_pickle(self, filename: str, filepath: str):
        """Save cache to a pickle file.

        Args:
            filename: Name of the output file
            filepath: Directory path to save to
        """
        data = {
            field: rules_list.get_data() for field, rules_list in self.fields.items()
        }

        # Ensure directory exists
        output_dir = os.path.dirname(os.path.join(filepath, filename))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(filepath, filename)
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        logger.info("Cache saved to pickle: %s", output_path)

    @classmethod
    def load_from_file_pickle(cls, filepath: str) -> "Cache":
        """Load cache from pickle file.

        Args:
            filepath: Path to the pickle file to load

        Returns:
            Cache instance populated with loaded rules
        """
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

        logger.info(
            "Cache loaded from pickle: %s (%d fields)",
            filepath,
            len(instance.fields),
        )
        return instance

    def __repr__(self):
        return f"Cache(fields={list(self.fields.keys())})"
