<template>
  <div>
    <div class="">
      <div class="flex flex-wrap">
        <p class="font-bold">Targets:</p>
        <div
          class="ml-2 underline"
          v-for="t in parsed_targets"
          :key="t.hash_id + 'name'"
        >
          {{ t.name }}
        </div>
      </div>
      <a-divider />
      <div>
        <p class="text-2xl my-5">Sequence</p>
        <MSAViewer :sequences="parsed_sequences" />
      </div>
      <a-divider />
      <div>
        <p class="text-2xl mt-5">Protein Viewer</p>
        <ProteinViewer :pdb_list="current_pdbs" />
      </div>
      <div>
        <p class="text-2xl my-5">MSA</p>
        <a-tabs v-model:activeKey="activeKey">
          <a-tab-pane
            v-for="t in parsed_targets"
            :key="t.hash_id"
            :tab="t.name"
            class="max-h-screen overflow-y-auto"
          >
            <!-- <div :id="'msa-viewer-' + t.hash_id"></div> -->
            <MSAViewer v-if="t.path_tree" :msa_path="t.path_tree.alphafold.input_a3m" />
          </a-tab-pane>
        </a-tabs>
      </div>
      <a-divider />
      <div>
        <p class="text-2xl my-5">MSA Coverage</p>
        <div class="flex flex-wrap justify-around">
          <div
            v-for="t in parsed_targets"
            :key="t.hash_id + 'msa'"
            class="flex flex-col items-center mb-5"
          >
            <img
              width="600"
              v-if="t.analysis"
              :src="t.analysis.msa_coverage"
              alt=""
            />
            <p class="text-gray-700">{{ t.name }}</p>
          </div>
        </div>
      </div>
      <a-divider />
      <div>
        <p class="text-2xl my-5">Predicted Distogram</p>
        <div>
          <div
            v-for="t in parsed_targets"
            :key="t.hash_id + 'distorgram'"
            class="flex flex-col items-center mb-5"
          >
            <img
              width="1000"
              v-if="t.analysis"
              :src="t.analysis.predict_distogram"
              alt=""
            />
            <p class="text-gray-700">{{ t.name }}</p>
          </div>
        </div>
      </div>
      <a-divider />
      <div>
        <div>
          <p class="text-2xl my-5">Predicted Contacts</p>
          <div
            v-for="t in parsed_targets"
            :key="t.hash_id + 'contacts'"
            class="flex flex-col items-center mb-5"
          >
            <img
              width="1000"
              v-if="t.analysis"
              :src="t.analysis.predict_contacts"
              alt=""
            />
            <p class="text-gray-700">{{ t.name }}</p>
          </div>
        </div>
      </div>
      <a-divider />
      <div>
        <p class="text-2xl my-5">Predicted LDDT</p>
        <div class="flex flex-wrap justify-around">
          <div
            v-for="t in parsed_targets"
            :key="t.hash_id + 'lddt'"
            class="flex flex-col items-center mb-5"
          >
            <img
              width="600"
              v-if="t.analysis"
              :src="t.analysis.predict_LDDT"
              alt=""
            />
            <p>{{ t.name }}</p>
          </div>
        </div>
      </div>
      <div>
        <div>
          <p class="text-2xl my-5">Conformation Analysis</p>
          <div
            v-for="t in parsed_targets"
            :key="t.hash_id + 'conformation'"
            class="flex flex-col items-center mb-5"
          >
            <div v-if="t.analysis">
              <div
                v-for="c in t.analysis.conformations"
                :key="t.hash_id + c.url"
                class="flex flex-col items-center"
              >
                <img width="640" :src="c.url" alt="" />
                <p class="text-xs text-gray-400">{{ c.name }}</p>
              </div>
            </div>
            <div>
              <p class="text-gray-700">{{ t.name }}</p>
              <p class="text-gray-400 text-xs" v-if="t.reserved && t.reserved.comment">
                ({{ t.reserved.comment }})
              </p>
            </div>
          </div>
        </div>
      </div>
      <a-divider />
    </div>
  </div>
</template>

<script>
import { map, split, keyBy, filter, forEach } from "lodash";
import { API } from "@/js/api";
import { Data } from "@/js/data";
import axios from "axios";

export default {
  name: "AnalysisView",
  components: {
  },

  mounted: function () {
    this.get_requests(() => {
      this.load_from_url();
      this.activeKey = this.parsed_targets[0].hash_id;
    });
  },

  data() {
    return {
      requests: [],
      requests_map: {},
      loading_requests: false,
      targets: [],
      current_view_pdb: null,
      psv: {},
      activeKey: "1",
      msa: {},
      psv_msa: {},
    };
  },

  methods: {
    load_from_url: function () {
      if (this.$route.query.hash_ids) {
        let hash_ids = split(this.$route.query.hash_ids, ",");
        this.targets = map(hash_ids, (hash_id) => {
          return this.requests_map[hash_id];
        });
      } else if (this.$route.query.name) {
        let names = split(this.$route.query.name, ",");
        this.targets = filter(this.requests, (r) => {
          for (let name of names) {
            if (r.name.toLowerCase().includes(name.toLowerCase())) {
              return true;
            }
          }
          return false;
        });
      }
    },

    get_requests: function (on_load) {
      this.loading_requests = true;
      axios(API.get_requests())
        .then((response) => {
          this.requests = Data.add_key(response.data);
          this.requests_map = keyBy(this.requests, "hash_id");
          if (on_load) {
            on_load();
          }
        })
        .catch((error) => {
          console.log(error);
        })
        .finally(() => {
          this.loading_requests = false;
        });
    },
  },

  computed: {
    current_pdbs() {
      let pdbs = [];
      forEach(this.targets, (record) => {
        forEach(Data.parse_pdbs_info(record), (pdb_info) => {
          pdbs.push({
            info: record,
            pdb_path: pdb_info.relaxed_pdb,
            name: record.name + "(" + pdb_info.model_name + ")",
            visible: pdb_info.relaxed_pdb.includes("rank_1"),
            plddt: pdb_info.plddt,
          });
        });
        if (record.reserved.exp_pdb_path) {
          pdbs.push({
            info: record,
            pdb_path: record.reserved.exp_pdb_path,
            name: record.name + "(GT)",
            plddt: null,
          });
        }
      });
      return pdbs;
    },
    parsed_targets() {
      return Data.parse_data(this.targets);
    },
    parsed_sequences() {
      return Data.parse_sequences(this.parsed_targets);
    },
  },
};
</script>