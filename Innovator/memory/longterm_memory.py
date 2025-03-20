from collections import defaultdict
from typing import Iterable
from docarray import DocList
from docarray.index import InMemoryExactNNIndex

from innovator.data import Info


class ShortTermMemory(DocList[Info]):
    """A memory store that maintains short-term information as DocList."""

    def add(self, info: Info) -> None:
        """Add a new Info object to memory if it is not already present."""
        if info not in self:
            self.append(info)

    def add_batch(self, infos: DocList[Info]) -> None:
        """Batch add multiple Info objects to memory, ensuring no duplicates."""
        existing_ids = {info.id for info in self}  # 使用 set 提高去重效率
        for info in infos:
            if info.id not in existing_ids:
                self.append(info)
                existing_ids.add(info.id)

    def remember(self, k: int = 0) -> DocList[Info]:
        """Retrieve the most recent k memories, return all when k <= 0."""
        return self[-max(0, k):] if k else self

    def remember_news(self, observed: DocList[Info], k: int = 0) -> DocList[Info]:
        """Retrieve the most recent k new observations that are not in memory."""
        already_observed_ids = {info.id for info in self.remember(k)}
        news = DocList[Info]([info for info in observed if info.id not in already_observed_ids])
        return news

    def remember_by_action(self, action: str) -> DocList[Info]:
        """Retrieve all Info objects that were triggered by a specific action."""
        return self._filter_by_actions([action])

    def remember_by_actions(self, actions: Iterable[str]) -> DocList[Info]:
        """Retrieve all Info objects that were triggered by specific actions."""
        return self._filter_by_actions(actions)

    def _filter_by_actions(self, actions: Iterable[str]) -> DocList[Info]:
        """Helper method to filter stored Info objects by given actions."""
        if not actions:
            return DocList[Info]()

        storage_index = InMemoryExactNNIndex[Info]()
        storage_index.index(self)

        filtered_contents = DocList[Info]()
        for action in actions:
            query = {'cause_by': {'$eq': action}}
            filtered_contents.extend(storage_index.filter(query))  # 使用 extend 确保类型一致

        return filtered_contents
