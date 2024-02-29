import dataclasses
from typing import List, Dict
from xmlrpc.client import Boolean


@dataclasses.dataclass(frozen=True)
class Structure:
    tag2idx: Dict[str, int]
    tags: List[str]
    atoms: List[List[str]]


class MMCIFParser:
    def __init__(self, mmcif_file) -> None:
        with open(mmcif_file, "r") as fd:
            self.blocks = fd.read().split("#")

    def get_all_block(self):
        return self.blocks

    def search_block(self, pattn):
        for block in self.blocks:
            if pattn in block:
                return block
        return None

    def _process_loop(self, loop_string: str):
        tags = []
        content = []
        for line in loop_string.split("\n"):
            item = line.strip()
            if line.startswith("loop_") or not item:
                continue
            if line.startswith("_"):
                tags.append(item)
            else:
                content.append(item)
        return tags, content
    
    def get_structure(self):
        struct_block = self.search_block("_atom_site.group_PDB")
        tags, content = self._process_loop(struct_block)
        
        tag2idx = dict(list(zip(tags, range(len(tags)))))
        
        return Structure(tag2idx=tag2idx, tags=tags, atoms=content)
    
    def get_chain(self, chain_id):
        st = self.get_structure()
        tag2idx = st.tag2idx
        tags = st.tags
        atoms = list(filter(lambda x: x.split()[st.tag2idx["_atom_site.auth_asym_id"]] == chain_id, st.atoms))
        return Structure(tag2idx=tag2idx, tags=tags, atoms=atoms)
    
    @staticmethod
    def write_structure(st: Structure, file_name: str, name: str = "tmp_target"):
        loop = [name, "#"]
        loop.append("loop_")
        loop.extend(st.tags)
        loop.extend(st.atoms)
        loop.append("#")
        with open(file_name, "w") as fd:
            fd.write("\n".join(loop))
        
      
    
        
        