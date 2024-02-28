<template>
  <div>
    <div>
      <a-button
        :loading="loading_default_conf"
        @click="update_view(config)"
        type="primary"
        ghost
      >
        Reset
      </a-button>
      <a-button
        class="mx-2"
        :loading="loading_default_conf"
        @click="load_default_conf"
        type="primary"
        ghost
      >
        Load Default Config
      </a-button>
      <div class="mt-5">
        <div class="font-bold">
          Copy from previous record ({{ this.config.name }}):
        </div>
        <div class="flex flex-wrap mt-2">
          <a-button class="mx-1" type="dashed" size="small" @click="set_msa"
            >msa</a-button
          >
          <a-button
            class="mx-1"
            type="dashed"
            size="small"
            @click="set_template"
            >template</a-button
          >
        </div>
      </div>
      <div class="mt-5" v-if="full_config.msa_select">
        <div class="font-bold">Set MSA Selection Strategy</div>
        <div class="flex flex-wrap mt-2">
          <a-button
            v-for="(params, strategy) in full_config.msa_select"
            class="mx-1"
            type="dashed"
            size="small"
            :key="strategy"
            @click="set_msa_select(strategy, params)"
            >{{ strategy }}</a-button
          >
        </div>
      </div>
    </div>
    <div id="editor_request" class="border mt-5"></div>

    <div class="flex justify-end">
      <div>
        <a-button
          :loading="submitting_requests"
          @click="submit_request"
          type="primary"
          ghost
          class="mt-5"
        >
          Submit
        </a-button>
      </div>
    </div>
  </div>
</template>

<script>
import { API } from "@/js/api";
import { Data } from "@/js/data";
import axios from "axios";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { defaultKeymap } from "@codemirror/commands";
import {
  syntaxHighlighting,
  defaultHighlightStyle,
} from "@codemirror/language";
import { json } from "@codemirror/lang-json";

export default {
  name: "Request",
  props: {
    request: Object,
    on_load: Function,
  },
  data() {
    return {
      submitting_requests: false,
      loading_default_conf: false,
      config: {},
      editor_request: {},
      full_config: {},
      checked: {},
      update_when_checked: true,
    };
  },
  mounted() {
    this.create_view(this.request);
    this.load_full_config();
  },
  methods: {
    submit_request() {
      this.submitting_requests = true;
      let request = API.submit_request(
        JSON.parse(this.editor_request.state.doc.toString())
      );
      axios(request)
        .then((response) => {
          if (this.on_load) {
            this.on_load(response);
          }
        })
        .finally(() => {
          this.submitting_requests = false;
        });
    },

    copy_config() {
      return JSON.parse(JSON.stringify(this.config));
    },

    config_from_editor() {
      return JSON.parse(this.editor_request.state.doc);
    },

    load_default_conf() {
      this.loading_default_conf = true;
      let config = this.config_from_editor();
      let mode = config.sender.includes("casp") ? "casp" : "cameo";
      axios(API.get_default_conf(mode))
        .then((response) => {
          config.run_config = response.data;
          this.editor_request.dispatch({
            changes: {
              from: 0,
              to: this.editor_request.state.doc.length,
              insert: this.config2text(config),
            },
          });
        })
        .finally(() => {
          this.loading_default_conf = false;
        });
    },

    load_full_config() {
      axios(API.get_default_conf("full"))
        .then((response) => {
          this.full_config = response.data;
        })
        .catch((error) => {
          console.log(error);
        });
    },

    config2text(config, renew) {
      if (renew) {
        this.config = JSON.parse(JSON.stringify(config));
        config = Data.renew_config(config);
      }
      let text = JSON.stringify(config, null, 2);
      return text;
    },

    modify_text(text, func) {
      let r = JSON.parse(text);
      if (func) {
        r = func(r);
      }
      this.editor_request.dispatch({
        changes: {
          from: 0,
          to: this.editor_request.state.doc.length,
          insert: this.config2text(r, false),
        },
      });
    },

    create_view(config) {
      let start_state = EditorState.create({
        doc: this.config2text(config, true),
        extensions: [
          keymap.of(defaultKeymap),
          syntaxHighlighting(defaultHighlightStyle),
          json(),
        ],
      });
      this.editor_request = new EditorView({
        state: start_state,
        parent: document.querySelector("#editor_request"),
      });
    },

    update_view(config) {
      this.editor_request.dispatch({
        changes: {
          from: 0,
          to: this.editor_request.state.doc.length,
          insert: this.config2text(config, true),
        },
      });
    },

    set_msa() {
      this.modify_text(this.editor_request.state.doc, (r) => {
        if (r.run_config) {
          if (r.run_config.msa_search) {
            r.run_config.msa_search.copy_int_msa_from =
              this.config.name + ".a3m";
          }
        }
        return r;
      });
    },

    set_template() {
      this.modify_text(this.editor_request.state.doc, (r) => {
        if (r.run_config) {
          if (r.run_config.template) {
            r.run_config.template.copy_template_hits_from =
              this.config.name + ".hits.pkl";
          }
        }
        return r;
      });
    },

    set_msa_select(strategy, params) {
      this.modify_text(this.editor_request.state.doc, (r) => {
        if (r.run_config) {
          if (r.run_config.msa_select) {
            if (strategy == "idle") {
              r.run_config.msa_select = {};
            } else {
              if ("idle" in r.run_config.msa_select) {
                delete r.run_config.msa_select.idle;
              }
            }
            r.run_config.msa_select[strategy] = params;
          }
        }
        return r;
      });
    }
  },

  watch: {
    request(new_config) {
      this.update_view(new_config);
    },

    checked: {
      handler(new_checked) {
        if (this.update_when_checked) {
          this.modify_text(this.editor_request.state.doc, (r) => {
            if (r.run_config) {
              if (r.run_config.msa_search) {
                r.run_config.msa_search.copy_int_msa_from = new_checked.msa
                  ? this.config.name + ".a3m"
                  : this.config.run_config.msa_search.copy_int_msa_from;
              }
              if (r.run_config.template) {
                r.run_config.template.copy_template_hits_from =
                  new_checked.template
                    ? this.config.name + ".hits.pkl"
                    : this.config.run_config.template.copy_template_hits_from;
              }
            }
            return r;
          });
        }
      },
      deep: true,
    },
  },
  computed: {},
};
</script>
