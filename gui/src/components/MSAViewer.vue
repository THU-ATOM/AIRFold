<template>
  <div>
    <div class="flex">
      <div>
        <a-radio-group v-model:value="filter_num">
          <a-radio-button value="all"
            >All ({{ loaded_sequences ? loaded_sequences.length : 0 }})</a-radio-button
          >
          <a-radio-button value="hundred">100</a-radio-button>
          <a-radio-button value="thound">1000</a-radio-button>
        </a-radio-group>
      </div>
      <div class="ml-2">
        <a-select v-model:value="sequence_color">
          <a-select-option
            v-for="color in sequence_colors"
            :value="color"
            :key="color"
            >{{ color }}</a-select-option
          >
        </a-select>
      </div>
    </div>
    <div :id="this.viewer_id"></div>
  </div>
</template>

<script>
import { Data } from "@/js/data";
import { cloneDeep } from "lodash";
// https://biocomputingup.github.io/ProSeqViewer-documentation/start
import { ProSeqViewer } from "proseqviewer/dist";
import "proseqviewer/dist/assets/proseqviewer.css";

export default {
  name: "MSAViewer",
  props: {
    id: {
      type: String,
      default: "",
    },
    msa_path: String,
    // {sequence: "", label: "", id: ""}
    sequences: {
      type: Array,
      default: () => [],
    },
  },
  data() {
    return {
      viewer: {},
      msa_id: "",
      loaded_sequences: [],
      filter_num: "hundred",
      sequence_color: "clustal",
      sequence_colors: [
        "blosum62",
        "clustal",
        // "zappo",
        // "taylor",
        // "hydrophobicity",
      ],
    };
  },
  mounted() {
    this.msa_id = this.msa_id || Data.makeid(8);
    this.load_msa();
  },
  methods: {
    load_msa: function () {
      if (this.msa_path) {
        Data.parse_msa(this.msa_path, (sequences) => {
          this.loaded_sequences = sequences ? sequences : [];
          this.filter_num =
            this.loaded_sequences.length > 100 ? "hundred" : "all";
        });
      } else {
        this.loaded_sequences = this.sequences;
        this.filter_num =
          this.loaded_sequences.length > 100 ? "hundred" : "all";
      }
    },

    render_msa: function () {
      if (this.view_sequences && this.view_sequences.length > 0) {
        document.getElementById(this.viewer_id).innerHTML = "";
        let viewer = new ProSeqViewer(this.viewer_id);
        viewer.draw({
          sequences: cloneDeep(this.view_sequences),
          options: this.options,
          consensus: this.consensus,
        });
      }
    },
  },
  watch: {
    view_sequences: function () {
      this.render_msa();
    },
    options: {
      deep: true,
      handler: function () {
        this.render_msa();
      },
    },
    sequences: function () {
      this.load_msa();
    },
    msa_path: function () {
      this.load_msa();
    },
  },
  computed: {
    view_sequences() {
      if (!this.loaded_sequences || this.loaded_sequences.length === 0) {
        return [];
      }
      if (this.filter_num === "all") {
        return this.loaded_sequences;
      } else if (this.filter_num === "hundred") {
        return this.loaded_sequences.slice(0, 100);
      } else if (this.filter_num === "thound") {
        return this.loaded_sequences.slice(0, 1000);
      }
      return [];
    },
    viewer_id() {
      return "msa-viewer-" + this.msa_id;
    },
    options() {
      return {
        chunkSize: 20,
        wrapLine: true,
        sequenceColor: this.sequence_color,
        indexesLocation: "top",
      };
    },
    consensus() {
      return { color: "physical", dotThreshold: 70 };
    },
  },
};
</script>
