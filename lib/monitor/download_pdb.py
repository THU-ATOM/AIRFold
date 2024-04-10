import requests, os
from datetime import datetime

# from threading import Thread
from multiprocessing import Process

from loguru import logger

from lib.monitor.info_report import InfoReport
from lib.constant import DB_PATH
import traceback

DATE = "date"
TARGET = "target"
NAME = "name"


class CameoPDBDownloader(Process):
    def __init__(
        self, exp_pdb_dir="/data/protein/CAMEO/exp_pdbs", to_date=None
    ) -> None:
        super(CameoPDBDownloader, self).__init__()
        if not os.path.exists(exp_pdb_dir):
            os.makedirs(exp_pdb_dir)
        self.cameo_manifest_url = "https://www.cameo3d.org/modeling/targets/1-week/ajax"
        self.pdb_base_url = "https://www.cameo3d.org/static/data/modeling"
        self.release_date = ""
        self._pdb_path = exp_pdb_dir
        self.info_reporter = InfoReport()
        if to_date:
            self._to_date = to_date
        else:
            self._to_date = datetime.now().strftime("%Y-%m-%d")

    def updated_pdb_manifest(self):
        res = requests.get(
            self.cameo_manifest_url,
            params={"to_date": f"{self._to_date}"},
        )
        ret = [item for item in res.json()["aaData"] if item[DATE] != self.release_date]
        if len(ret) > 0:
            self.release_date = ret[0][DATE]
        return ret

    # {'target': '2022-04-23_00000006_1', 'pdbid': '7ctx', 'pdbid_chain': 'B', 'pred_num': 54, 'oligo_pred_num': 0, 'ass_oligo': 0, 'ass_heterooligo': 0, 'ass_qs_state': 0, 'ass_lig': {'I': None, 'O': None, 'N': None, 'P': None}, 'ref_pdb': '7ctx [B]', 'seq_length': 323, 'exp': 'X-RAY DIFFRACTION', 'res': 2.907, 'diff': '1', 'date': '2022-04-23'}
    @staticmethod
    def download_file(url, file_path):
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)

    def download_pdbs(self, manifest):
        for item in manifest:
            pdb_id = f"{item['pdbid'].upper()}_{item['pdbid_chain'].upper()}"
            fpath = os.path.join(self._pdb_path, f"{pdb_id}.pdb")
            if os.path.exists(fpath):
                logger.warning(f"file {fpath} already exists")
            else:
                url = "/".join(
                    [
                        self.pdb_base_url,
                        item[DATE].replace("-", "."),
                        pdb_id,
                        "target.pdb",
                    ]
                )
                logger.info(f"downloading {url} to {fpath}")
                self.download_file(url, fpath)
            target = item[TARGET]
            hash_ids = self.info_reporter.get_hash_ids(query_dict={NAME: f"{target}%"})
            for hash_id in hash_ids:
                self.info_reporter.update_reserved(
                    hash_id=hash_id, update_dict={"exp_pdb_path": fpath}
                )
                try:
                    self.info_reporter.update_lddt_metric(hash_id=hash_id)
                except:
                    traceback.print_exc()
                    logger.error("udpate lddt metric failed")
            logger.info(f"{target}: {fpath} downloaded and updated the mapping")

    def run(self) -> None:
        manifests = self.updated_pdb_manifest()
        if len(manifests) > 0:
            logger.info(f"find {len(manifests)} new pdbs.")
            self.download_pdbs(manifest=manifests)
