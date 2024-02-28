<template>
  <div class="px-5">
    <div class="mt-5 flex justify-between">
      <div class="flex justify-between items-center mt-5">
        <div class="w-96 border-b flex items-center">
          <input
            v-model="filter_name"
            class="
              placeholder:italic placeholder:text-slate-400
              bg-white
              w-full
              py-2
              pl-2
              pr-3
              focus:outline-none
              sm:text-sm
            "
            placeholder="Comparing tags, separated with ,"
            type="text"
            name="search"
          />
          <close-outlined @click="filter_name = ''" class="cursor-pointer" />
        </div>

        <a-button
          class="ml-5"
          @click="draw_table"
          type="primary"
          :disabled="loading_requests"
          ghost
        >
          Compare
        </a-button>
      </div>
      <div>
        <div class="flex items-center mt-3">
          <span class="mr-2 text-xs text-gray-400">
            {{
              "Last refresh: " +
              moment(last_refresh).format("YYYY-MM-DD HH:mm:ss")
            }}
          </span>
          <a-tooltip placement="topRight">
            <template #title>
              <span>Auto Refresh per {{ refresh_interval }} seconds.</span>
            </template>
            <a-switch
              class="m-1"
              v-model:checked="is_auto_refresh"
              size="default"
            />
          </a-tooltip>
          <div
            @click="get_requests"
            class="cursor-pointer m-1 flex items-center text-lg"
          >
            <loading-outlined v-if="loading_requests" />
            <sync-outlined v-else />
          </div>
        </div>
      </div>
    </div>

    <div class="mt-2">
      <a-switch v-model:checked="share_only" size="default" />
      Compare shared results only
    </div>
    <div class="flex items-center mt-5 flex-wrap">
      <a-tag
        v-for="tag in available_tags"
        class="cursor-pointer m-1"
        color="blue"
        :key="tag"
        @click="toggle_tag(tag)"
        >{{ tag }}</a-tag
      >
    </div>

    <div>
      <!-- <div class="mt-5">Total samples: {{ this.detail.length }}</div> -->

      <a-table
        class="mt-5"
        :dataSource="this.statistics"
        :columns="statistics_columns"
        size="middle"
        :pagination="{ defaultPageSize: 100 }"
      >
      </a-table>
    </div>
  </div>
</template>

<script>
import { API } from "@/js/api";
import { Data } from "@/js/data";
import axios from "axios";
import { forOwn, keyBy, forEach } from "lodash";
import {
  SyncOutlined,
  LoadingOutlined,
  CloseOutlined,
} from "@ant-design/icons-vue";
import moment from "moment";

export default {
  name: "CompareView",
  components: {
    SyncOutlined,
    LoadingOutlined,
    CloseOutlined,
  },

  mounted: function () {
    this.get_requests();
    this.today = moment();
    setInterval(() => {
      if (
        this.is_auto_refresh &&
        !document.hidden &&
        moment().diff(this.last_refresh, "seconds") > this.refresh_interval
      ) {
        this.get_requests();
        this.today = moment();
      }
    }, 10000);
  },

  data() {
    return {
      // data
      requests: [],
      requests_map: {},
      selected_hash_ids: [],
      filter_date: "month",
      filter_sender: "casp",
      filter_name: "",
      hide_invisible: true,
      cameo_data: {},
      casp_data: {},
      today: {},

      statistics: [],
      detail: [],
      share_only: false,

      // state
      loading_rerun: false,
      loading_requests: false,
      loading_email: false,
      loading_lddt: false,
      show_comment: true,
      comment_edit_visible: false,
      current_comment_record: {},
      current_comment: "",
      is_auto_refresh: true,
      last_refresh: {},
      refresh_interval: 120, // seconds
      max_show_history: 48,
      loading_cameo: false,
      request_visible: false,
      current_request_json: {},
      protein_visible: false,
      current_view_pdb: null,
    };
  },

  computed: {
    has_selected() {
      return this.selected_hash_ids.length > 0;
    },

    parsed_data() {
      return Data.parse_data(this.requests);
    },

    available_tags() {
      let tags = [];
      forEach(this.parsed_data, (record) => {
        if (record.tags) {
          forEach(record.tags, (tag) => {
            if (tags.indexOf(tag) < 0) {
              tags.push(tag);
            }
          });
        }
      });
      return tags;
    },

    statistics_columns() {
      let tags = this.filter_name.split(",");
      let columns = [
        {
          title: "Metric",
          dataIndex: "metric",
          key: "metric",
          sorter: (a, b) => a.metric.localeCompare(b.metric),
          defaultSortOrder: "ascend",
        },
      ];
      forEach(tags, (tag) => {
        columns.push({
          title: tag,
          dataIndex: tag,
          key: tag,
          sorter: (a, b) => a[tag] > b[tag],
        });
      });
      return columns;
    },
  },

  methods: {
    moment,
    // APIs
    get_requests: function () {
      this.loading_requests = true;
      axios(API.get_requests())
        .then((response) => {
          this.requests = Data.add_key(response.data);
          this.requests_map = keyBy(this.requests, "hash_id");
          this.last_refresh = new Date();
        })
        .catch((error) => {
          console.log(error);
        })
        .finally(() => {
          this.loading_requests = false;
        });
    },

    toggle_tag: function (tag) {
      let tags = [];
      let tag_exists = false;
      if (this.filter_name) {
        forEach(this.filter_name.split(","), (t) => {
          if (t.trim() !== tag) {
            tags.push(t.trim());
          } else {
            tag_exists = true;
          }
        });
      }
      if (!tag_exists) {
        tags.push(tag);
      }
      this.filter_name = tags.join(",");
    },

    remove_tag: function (tag) {
      let tags = [];
      if (this.filter_name) {
        forEach(this.filter_name.split(","), (t) => {
          if (t.trim() !== tag) {
            tags.push(t.trim());
          }
        });
      }
      this.filter_name = tags.join(",");
    },

    draw_table: function () {
      let raw_data = this.filter_raw_data();
      let result = this.compare_data(raw_data);
      this.detail = this.object2array(result.data, "target");
      this.statistics = this.object2array(
        this.merge_delta(result.statistics),
        "metric"
      );
    },

    filter_raw_data: function () {
      let tags = this.filter_name.split(",");
      let table_raw = {};
      forEach(this.parsed_data, (record) => {
        let target = record.request_json.target;
        let record_tags = record.tags;
        forEach(record_tags, (tag) => {
          if (tags.includes(tag)) {
            if (!table_raw[target]) {
              table_raw[target] = {};
            }
            table_raw[target][tag] = Data.parse_pdbs_info(record, true);
          }
        });
      });
      return table_raw;
    },
    compare_data: function (raw_table) {
      let tags = this.filter_name.split(",");
      let primary_tag = tags[0];
      let target_keys = ["plddt", "lddt"];
      let overall = {};
      let count = {};
      // sequence, each seq contains multiple experimental results
      forOwn(raw_table, (exps) => {
        // when share_only is true, only compare full exps
        // console.log(this.share_only);
        if (this.share_only) {
          for (let i = 0; i < tags.length; i++) {
            if (!exps[tags[i]]) {
              return;
            }
          }
        }

        // get primary exp & rank_1_primary
        let primary = exps[primary_tag];
        let primary_models = {};
        let rank_1_primary = null;
        if (primary) {
          forEach(primary, (model) => {
            primary_models[model.model_name] = model;
            if (model["rank"] == 1) {
              rank_1_primary = model;
            }
          });
        }
        // iterate all exps, exp: a list of model results
        forOwn(exps, (models, tag) => {
          if (!overall[tag]) {
            overall[tag] = new Proxy(
              {},
              {
                get: (target, name) => (name in target ? target[name] : 0),
              }
            );
            count[tag] = new Proxy(
              {},
              {
                get: (target, name) => (name in target ? target[name] : 0),
              }
            );
          }
          overall[tag]["sample"] += 1;
          // iterate all models of this exp
          let exp_models = {};
          let rank_1_exp = null;
          forEach(models, (exp_model) => {
            overall[tag]["model"] += 1;
            let model_name = exp_model.model_name;
            exp_models[model_name] = exp_model;
            forEach(target_keys, (key) => {
              if (exp_model[key]) {
                overall[tag][key] += exp_model[key];
                count[tag][key] += 1;
                overall[tag][model_name + "_" + key] += exp_model[key];
                count[tag][model_name + "_" + key] += 1;
              }
            });
            if (exp_model["rank"] == 1) {
              rank_1_exp = exp_model;
            }
          });

          // compare models with primary, m1 to m1, m2 to m2, ...
          forOwn(exp_models, (exp_model, model_name) => {
            let primary_model = primary_models[model_name];
            if (primary_model && primary_model !== exp_model) {
              forEach(target_keys, (key) => {
                if (primary_model[key] && exp_model[key]) {
                  exp_model["d_" + key] = exp_model[key] - primary_model[key];
                  overall[tag]["d_" + model_name + "_" + key] +=
                    exp_model["d_" + key];
                  count[tag]["d_" + model_name + "_" + key] += 1;
                  overall[tag]["d_" + key] += exp_model["d_" + key];
                  count[tag]["d_" + key] += 1;
                }
              });
            }
          });

          // compare rank_1 to rank_1
          if (rank_1_primary && rank_1_exp) {
            forEach(target_keys, (key) => {
              if (rank_1_exp[key]) {
                overall[tag]["rank_1_" + key] += rank_1_exp[key];
                count[tag]["rank_1_" + key] += 1;
              }
              if (rank_1_primary[key] && rank_1_exp[key]) {
                rank_1_exp["d_" + key] = rank_1_exp[key] - rank_1_primary[key];
                overall[tag]["d_" + "rank_1_" + key] += rank_1_exp["d_" + key];
                count[tag]["d_" + "rank_1_" + key] += 1;
              }
            });
          }
        });
      });
      let mean = {};
      forOwn(overall, (items, tag) => {
        mean[tag] = {};
        forOwn(items, (value, key) => {
          if (count[tag][key]) {
            mean[tag][key] = value / count[tag][key];
            // console.log(tag, key, value, count[tag][key]);
          } else {
            mean[tag][key] = value;
          }
        });
      });
      return {
        data: raw_table,
        statistics: {
          overall: this.transform_table(overall),
          count: this.transform_table(count),
          mean: this.transform_table(mean),
        },
      };
    },

    transform_table: function (table) {
      let result = {};
      forOwn(table, (items, row) => {
        forOwn(items, (item, column) => {
          if (!result[column]) {
            result[column] = {};
          }
          result[column][row] = item;
        });
      });
      return result;
    },

    merge_delta: function (tables) {
      let table = tables.mean;
      let result = {};
      forOwn(table, (items, name) => {
        if (!name.startsWith("d_")) {
          result[name] = {};
          let delta_key = "d_" + name;
          let delta_obj = table[delta_key];
          let count_delta_obj = tables.count[delta_key];
          forOwn(items, (item, key) => {
            if (delta_obj && delta_obj[key]) {
              result[name][key] =
                "" +
                item.toFixed(2) +
                " (" +
                tables.count[name][key] +
                "), " +
                delta_obj[key].toFixed(2) +
                " (" +
                count_delta_obj[key] +
                ")";
            } else if (name == "sample" || name == "model") {
              result[name][key] = "" + item;
            } else {
              result[name][key] =
                "" + item.toFixed(2) + " (" + tables.count[name][key] + ")";
            }
          });
        }
      });
      return result;
    },

    object2array: function (obj, key_name) {
      let result = [];
      forOwn(obj, (item, key) => {
        item[key_name] = key;
        result.push(item);
      });
      return result;
    },
  },
};
</script>

<style>
.ant-switch {
  background-color: #d9d9d9 !important;
}

.ant-switch-checked {
  background-color: #1890ff !important;
}
</style>
