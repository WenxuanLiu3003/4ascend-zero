# 4ascend Game Rules 

## I. Basic Setup
- **Board**: 9×9 grid.
- **Players**: Two sides, Black and White, taking turns to place stones.
- **Health Points (HP)**: Both players start with an initial HP (recommended 20, adjustable by agreement).
- **Power**: Stones that form a chain of four or more during play will convert into offensive/defensive power upon resolution.

## II. Chains and Counting
- **Trigger Condition**: When either player forms a chain of **four or more stones in a row** (horizontal, vertical, or diagonal) after placing a stone, the game immediately enters the **Attack–Defense Phase**.
- **Multiple Chains**: Multiple chains may be triggered in the same turn. The total power is counted as the **number of unique stones involved across all chains**, avoiding duplicate counting.
- **Five-in-a-row or more**: Count the actual number of stones involved (e.g., a five-in-a-row counts as 5 power).
- **Chain Removal**: All stones forming the triggered chains (for either the attacker or the defender if they also form a chain in response) are removed from the board. The removed positions become empty and can be occupied by the opponent in the same turn (see Attack–Defense Phase).

## III. Plants (Resource Tiles)
- **Regeneration**: During the game, **plants will randomly appear on unoccupied tiles**. Additionally, **after each Attack–Defense resolution**, new plants are **immediately regenerated** in random empty tiles (quantity and frequency can be customized).
- **Effect**: If a player’s chain includes tiles containing plants, those plants will grant the player **+1 Power per plant** during the resolution phase. Whether a plant contributes depends on whether the tile remains **controlled/occupied by that player** at resolution time.

## IV. Turn Cycle and Attack–Defense Flow
1. **Normal Phase**: Players alternately place stones on empty tiles.
2. **Entering Attack–Defense Phase**: The player who first forms a chain of four or more becomes the **Attacker** and immediately removes all stones in their chains (these “removed positions” are now empty).
3. **Defender’s Turn**:
   - The Defender must **immediately place a stone**; it may be placed on any empty tile, including those just vacated by the Attacker’s removed stones. (If placed on such a tile, that position is considered “occupied by the Defender” for resolution.)
   - If this move also forms a chain of four or more, the Defender’s stones are also removed, and those removed positions are included in resolution.
4. **Unified Power Calculation**:
   - **Attacker’s Power** = the number of the Attacker’s removed stone positions that **remain unoccupied by the Defender at resolution**, **plus** the number of **plants on those same unoccupied positions**.
   - **Defender’s Power** = the number of the Defender’s removed stones **plus** the number of **plants among those removed positions**. (Since the Defender plays second, their removed positions cannot be occupied by the opponent.)
   - Note: If the Defender does not form a chain, their “removed stones” and “plants” count as 0, i.e. **Defender’s Power = 0**.
5. **HP Resolution**: Both sides’ powers offset each other. The side with less power loses HP equal to the **difference** in power.
6. **Exit Attack–Defense Phase**: After resolution, the board remains with the removed stones cleared. **Immediately regenerate new plants** on random empty tiles, then return to the Normal Phase and continue alternating turns.

> **Remark:** With this unified resolution rule, the intuitive effect of “nullification” is naturally represented:
> - If the Defender places a stone on a tile previously occupied by the Attacker’s removed stones, that tile **no longer contributes to the Attacker’s Power**.
> - If that tile originally contained a plant, the Attacker **loses that plant’s bonus** as well.
> - Meanwhile, if the Defender forms a chain and removes stones, **all plants within the Defender’s removed positions count toward their Power**.

## V. Victory Condition
- When either player’s HP ≤ 0, that player immediately loses, and the opponent wins.

## VI. Recommended Adjustable Parameters
- Initial HP (e.g., 15 / 20 / 30).
- Number and frequency of plant regeneration each round (fixed or within a range).
- Optional Draw Rule: If the maximum number of turns is reached and neither side is defeated, the winner may be determined by remaining HP or declared a draw.

## VII. Implementation Notes (for Developers/Referees)
- **Unique Counting**: If a single stone belongs to multiple chains, count its contribution only once.
- **Legal Moves**: Any empty tile is a legal move; during the Attack–Defense Phase, the Attacker’s removed positions are also considered empty and valid for the Defender’s move.
- **Randomness Source**: The only randomness in the game comes from plant regeneration; all other rules are deterministic.
