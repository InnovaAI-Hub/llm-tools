"""
Description:
    In this file we define the role of the message.

Classes:
    - MsgRoleType: A class for storing the role of the message.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 13.10.2024
Version: 0.1
Python Version: 3.10

TODO:
    Need to refactor this class. Goal it's define one type for user,
    that will be latter convert to correct models role.
"""

from enum import StrEnum


class MsgRoleType(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
