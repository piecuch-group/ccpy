import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

FORTRAN_INCLUDE_PAT = r"^\s*include\s*['\"](\w+\.\w+)['\"]"
FORTRAN_MODULE_PAT = r"^\s*\bmodule\b\s+(\w+)\s*(?:!+.*)*$"
FORTRAN_SUBMOD_PAT = r"^\s*\bsubmodule\b\s*\((\w+:?\w+)\)\s*(\w+)"
FORTRAN_USE_PAT = r"^\s*use,?\s*(?:non_intrinsic)?\s*(?:::)?\s*(\w+)"

FORTRAN_MODULE_RE = re.compile(FORTRAN_MODULE_PAT, re.IGNORECASE)
FORTRAN_SUBMOD_RE = re.compile(FORTRAN_SUBMOD_PAT, re.IGNORECASE)
FORTRAN_USE_RE = re.compile(FORTRAN_USE_PAT, re.IGNORECASE)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DependencyError(Exception):
    """Base class for dependency-related errors."""


class ExportExistsError(DependencyError):
    """Raised when a module is exported multiple times."""


class NeedNotExported(DependencyError):
    """Raised when a required module is not exported."""


def _has_cycle(graph: Dict[str, Set[str]]) -> Optional[List[str]]:
    """Detect cycles in the dependency graph."""
    visited = set()
    path = []

    def visit(vertex: str) -> Optional[List[str]]:
        if vertex in path:
            cycle_start = path.index(vertex)
            return path[cycle_start:] + [vertex]
        if vertex in visited:
            return None

        visited.add(vertex)
        path.append(vertex)

        for neighbor in graph[vertex]:
            if cycle := visit(neighbor):
                return cycle

        path.pop()
        return None

    for vertex in graph:
        if cycle := visit(vertex):
            return cycle
    return None


class DependencyScan:

    def __init__(self, input_files: List[str]):
        self.input_files = [Path(input_file) for input_file in input_files]
        self.exports: Dict[str, str] = dict()
        self.needs: Dict[str, Set[str]] = defaultdict(set)
        self.needs_mod: Dict[str, Set[str]] = defaultdict(set)

    def scan_file(self, input_file: Path) -> None:
        """
        Scan a single Fortran file for module definitions and dependencies.

        Args:
            input_file: Path to the Fortran source file
        Raises:
            ExportExistsError: If a module is defined multiple times
        """
        file_lines = input_file.read_text().splitlines()

        file_str = str(input_file)
        for line in file_lines:
            module_m = FORTRAN_MODULE_RE.match(line)
            use_m = FORTRAN_USE_RE.match(line)

            if module_m:
                exported_module = module_m.group(1).lower()
                if exported_module in self.exports:
                    raise ExportExistsError(
                        f"Module '{exported_module}' already exported"
                    )

                self.exports[exported_module] = file_str
                logger.debug(f"Found module '{exported_module}' in '{file_str}'")

            if use_m:
                needs_module = use_m.group(1).lower()
                self.needs_mod[file_str].add(needs_module)
                logger.debug(f"Found dependency: '{file_str}' needs '{needs_module}'")

    def scan(self) -> None:
        """Scan all input files and build dependency mappings."""
        for file in self.input_files:
            self.scan_file(file)

        self._map_needs_to_files()

        graph = {str(f): set() for f in self.input_files}
        graph.update(self.needs)
        if cycle := _has_cycle(graph):
            cycle_str = " -> ".join(cycle)
            raise DependencyError(f"Circular dependency detected: {cycle_str}")

    def _map_needs_to_files(self) -> None:
        """Map module dependencies to their corresponding files."""
        for file, needs in self.needs_mod.items():
            for need in needs:
                if need in self.exports:
                    self.needs[file].add(self.exports[need])

    def order_dependencies(self) -> List[str]:
        """
        Order files based on their dependencies.

        Returns:
            List of file paths in dependency order
        """
        output_order = [str(file) for file in self.input_files]

        while True:
            moved = False
            for file in self.input_files:
                file_str = str(file)
                file_pos = output_order.index(file_str)

                for need in self.needs[file_str]:
                    need_pos = output_order.index(need)
                    if need_pos > file_pos:
                        output_order[need_pos] = file_str
                        output_order[file_pos] = need
                        file_pos = need_pos
                        moved = True

            if not moved:
                break

        return output_order


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        logger.error(
            "Usage: order_fortran_dependencies.py <fortran_file1> <fortran_file2> ..."
        )
        sys.exit(1)

    files = sys.argv[1:]
    scan = DependencyScan(files)
    scan.scan()
    ordered_files = scan.order_dependencies()
    print(" ".join(ordered_files))


if __name__ == "__main__":
    main()
