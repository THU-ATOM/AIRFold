<template>
  <div>
    <div ref="figure">
      <div class="relative">
        <div
          ref="ruler_x"
          class="absolute ruler_x bottom-0 z-50"
          v-bind:class="{
            'opacity-0': !this.ruler_visible,
          }"
          v-bind:style="{
            height: this.ruler_width + 'px',
            width: this.img_width - this.ruler_width + 'px',
            left: this.ruler_width + 'px',
          }"
        ></div>
        <div
          ref="ruler_y"
          class="absolute ruler_y h-full w-6 z-40"
          v-bind:class="{
            'opacity-0': !this.ruler_visible,
          }"
          v-bind:style="{
            width: this.ruler_width + 'px',
          }"
        ></div>
        <img
          v-bind:style="{
            padding: this.ruler_width + 'px',
          }"
          @load="refresh"
          ref="img"
          :src="src"
          :width="img_width"
          alt=""
        />
      </div>
    </div>
    <div class="flex mt-2 items-center justify-between">
      <div
        class="flex items-center text-sm"
        v-bind:class="{
          'opacity-0': !this.ruler_visible,
        }"
      >
        <div class="ml-3">
          <span class="cursor-pointer" @click="set_last_guides"
            >Ruler Num:
          </span>
          <input
            ref="num_ruler"
            class="outline-none border px-2 ml-2 w-16"
            v-model="num_ruler"
          />
        </div>
        <div class="ml-3">
          <span @click="set_max_marker" class="cursor-pointer"
            >Marker Num:
          </span>
          <input
            ref="num_plot"
            class="outline-none border px-2 ml-2 w-16"
            v-model="num_plot"
          />
        </div>
        <a-button size="small" class="ml-3" @click="calibrate"
          >Calibrate</a-button
        >
        <a-button size="small" class="ml-3" @click="copy_labels"
          >Copy Labels</a-button
        >
      </div>
      <div class="flex">
        <column-width-outlined
          class="cursor-pointer hover:text-gray-700 mr-2"
          @click="ruler_visible = !ruler_visible"
          v-bind:class="{
            'text-gray-300': !ruler_visible,
            'text-gray-700': ruler_visible,
          }"
        />
        <a-popover placement="top">
          <template #content>
            <div class="text-xs text-gray-500">
              <p>1. Drag the horizontal ruler so that the first guideline is aligen to 0 marker.</p>
              <p>Repeat untill the coveragence:</p>
              <p>
                &nbsp; 2. Drag the last guideline to align the "Mark Num".
              </p>
              <p>
                &nbsp; 3. Mouse click the label "Ruler Num".
              </p>
              <p>&nbsp; 4. Click "Calibrate".</p>
              <p>5. Drag guidlines as you want click "Copy Labels" to get numbers.</p>
            </div>
          </template>
          <question-circle-outlined />
        </a-popover>
      </div>
    </div>
  </div>
</template>

<script>
import Guides from "@scena/guides";
import Ruler from "@scena/ruler";
import { drag } from "@daybrush/drag";
import {
  ColumnWidthOutlined,
  QuestionCircleOutlined,
} from "@ant-design/icons-vue";
import { message } from "ant-design-vue";

export default {
  name: "GuidesFigure",
  components: {
    ColumnWidthOutlined,
    QuestionCircleOutlined,
  },
  props: {
    src: {
      type: String,
      default: "",
    },
    img_width: {
      type: Number,
      default: 600,
    },
    ruler_width: {
      type: Number,
      default: 16,
    },
    init_ruler_offset: {
      type: Number,
      default: 0,
    },
    init_num_plot: {
      type: Number,
      default: 1,
    },
    init_num_ruler: {
      type: Number,
      default: 1,
    },
    ruler_visible: {
      type: Boolean,
      default: true,
    },
    max_num: {
      type: Number,
      default: Infinity,
    },
  },
  data() {
    return {
      ruler_x: null,
      guides_y: null,
      num_plot: 1,
      num_ruler: 1,
      last_scroll: 0,
      drag_dist: 0,

      unit: 50,
      short_line_size: 3.5,
      long_line_size: 5,
      main_line_size: "50%",
    };
  },
  mounted() {
    this.num_plot = this.init_num_plot;
    this.num_ruler = this.init_num_ruler;
    this.show_ruler();
    this.set_max_marker();
    this.set_last_guides();
  },
  methods: {
    show_ruler: function (e) {
      let ruler_x_elem = this.$refs.ruler_x;
      let ruler_y_elem = this.$refs.ruler_y;
      ruler_x_elem.innerHTML = "";
      ruler_y_elem.innerHTML = "";

      this.ruler_x = new Ruler(ruler_x_elem, {
        type: "horizontal",
        zoom: this.zoom,
        unit: this.unit,
        mainLineSize: this.main_line_size,
        longLineSize: this.long_line_size,
        shortLineSize: this.short_line_size,
        textOffset: [0, this.ruler_width - this.short_line_size],
      });
      this.guides_y = new Guides(ruler_y_elem, {
        type: "vertical",
        displayDragPos: true,
        zoom: this.zoom,
        defaultGuides: [0, this.img_width * 0.5],
        unit: this.unit,
        mainLineSize: this.main_line_size,
        longLineSize: this.long_line_size,
        shortLineSize: this.short_line_size,
        textOffset: [this.ruler_width - this.short_line_size, 0],
        dragGuideStyle: {
          color: "#0f0",
        },
        guideStyle: {
          border: "0.5px #0f0 solid",
        },
      });
      this.scroll(this.init_ruler_offset);
      this.last_scroll = this.init_ruler_offset;
      this.dragger = drag(this.$refs.ruler_x, {
        container: window,
        dragstart: ({ inputEvent }) => {
          inputEvent.stopPropagation();
          this.drag_dist = 0;
        },
        drag: ({ distX, distY }) => {
          this.scroll(this.last_scroll + distX);
          this.drag_dist = distX;
        },
        dragend: () => {
          this.last_scroll += this.drag_dist;
        },
      });
    },
    refresh: function () {
      this.ruler_x.resize();
      this.guides_y.resize();
    },
    scroll: function (x) {
      this.ruler_x.scroll(-x);
      this.guides_y.scrollGuides(-x);
    },
    calibrate: function () {
      let zoom = this.zoom * this.ruler_x.zoom;
      this.ruler_x.unit = this.possible_unit;
      this.guides_y.unit = this.possible_unit;
      this.ruler_x.zoom = zoom;
      this.guides_y.zoom = zoom;

      this.last_scroll /= this.zoom;
      // I am confused here, you must scroll the exact value minus a very little,
      // number, or it scroll to a nonsense position. Maybe it hits some boundary
      // condition.
      this.scroll(this.last_scroll - 0.0001);

      let guides = this.guides_y.getGuides();
      for (let i = 0; i < guides.length; i++) {
        guides[i] /= this.zoom;
      }
      this.guides_y.loadGuides(guides);
    },
    copy_labels: function () {
      let labels = [];
      let guides = this.guides_y.getGuides();
      for (let i = 0; i < guides.length; i++) {
        let x = Math.max(0, Math.min(guides[i], this.max_num));
        labels.push(x);
      }
      this.$copyText(labels.toString()).then(
        function (e) {
          message.success("Copied to clipboard.");
        },
        function (e) {
          message.warning("Copy failed.");
        }
      );
    },
    set_last_guides: function () {
      let guides = this.guides_y.getGuides();
      if (guides.length > 0) {
        let last_guide = guides[guides.length - 1];
        this.num_ruler = last_guide;
      }
    },
    set_max_marker: function () {
      if (this.max_num < Infinity) {
        let max_marker = this.max_num - (this.max_num % this.possible_unit);
        this.num_plot = max_marker;
      }
    },
  },
  computed: {
    zoom: function () {
      return this.num_ruler / this.num_plot;
    },
    possible_unit: function () {
      let possible_units = [
        5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000,
      ];
      let max_segs = 9;
      let final_unit = this.unit;
      if (this.max_num < Infinity) {
        for (let i = 0; i < possible_units.length; i++) {
          if (this.max_num / possible_units[i] < max_segs) {
            final_unit = possible_units[i];
            break;
          }
        }
      }
      return final_unit;
    },
  },
  watch: {},
};
</script>
