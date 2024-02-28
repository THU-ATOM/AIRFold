<template>
  <div>
    <div class="flex justify-center">
      <div class="w-full">
        <div id="protein_viewer"></div>
        <div id="picked-atom-name"></div>
      </div>
    </div>
    <div class="mt-5">
      <div class="flex justify-between">
        <div class="flex">
          <a-button class="ml-2" @click="clear_all">Clear</a-button>
          <a-button
            :disabled="viewing_pdb_keys.length < 2"
            @click="align_all"
            class="ml-2"
            >Align</a-button
          >
          <a-button
            :disabled="viewing_pdb_keys.length < 2"
            @click="tmalign_all"
            :loading="tm_aligning"
            class="ml-2"
            >TMAlign</a-button
          >
          <a-select class="ml-2" v-model:value="mode" :loading="rendering">
            <a-select-option
              v-for="m in modes"
              :value="m.value"
              :key="m.value"
              >{{ m.name }}</a-select-option
            >
          </a-select>
          <!-- <div class="ml-2">
            <loading-outlined v-if="rendering" />
            <sync-outlined v-else />
          </div> -->
        </div>
        <div>
          Show Residue:
          <a-switch
            class="pl-2"
            v-model:checked="show_residue"
            size="default"
          />
        </div>
      </div>
      <div class="mt-5">
        <a-table
          :dataSource="pdbs"
          :columns="columns"
          :row-selection="{
            selectedRowKeys: viewing_pdb_keys,
            onChange: on_select_change,
          }"
        >
          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'name'">
              {{ record.name ? record.name : record.info.name }}
            </template>
            <template v-if="column.key === 'color'">
              <p
                class="w-10 h-6 cursor-pointer"
                :style="'background-color:' + record.color + ';'"
                @click="change_color(record)"
              ></p>
            </template>
            <template v-if="column.key === 'action'">
              <delete-outlined @click="remove_pdb(record)" />
              <download-outlined class="ml-2" @click="download_pdb(record)" />
            </template>
          </template>
        </a-table>
      </div>
    </div>
    <div class="mt-5">
      <div class="flex justify-around flex-wrap">
        <div class="max-h-96" ref="heatmap_tmscore"></div>
        <div class="max-h-96" ref="heatmap_rmsd"></div>
      </div>
    </div>
  </div>
</template>

<script>
import { API } from "@/js/api";
import {
  DeleteOutlined,
  DownloadOutlined,
  SyncOutlined,
  LoadingOutlined,
} from "@ant-design/icons-vue";
import { forEach, split, map, cloneDeep } from "lodash";
import pv from "bio-pv";
import randomColor from "randomcolor";
import { message } from "ant-design-vue";
import axios from "axios";
import Plotly from "plotly.js-dist/plotly";

export default {
  name: "ProteinViewer",
  components: {
    DeleteOutlined,
    DownloadOutlined,
    SyncOutlined,
    LoadingOutlined,
  },
  props: {
    pdb: Object,
    pdb_list: Array,
    width: {
      default: "auto",
    },
    height: {
      type: Number,
      default: 600,
    },
  },
  data() {
    return {
      viewer: {},
      pdbs: [],
      viewing_pdb_keys: [],
      prevPicked: null,
      dragging: false,
      show_residue: false,
      rendering: false,
      tm_aligning: false,

      mode: "cartoon",
      modes: [
        {
          value: "sline",
          name: "Smooth Line Trace",
        },
        {
          value: "lines",
          name: "Lines",
        },
        {
          value: "trace",
          name: "Trace",
        },
        {
          value: "lineTrace",
          name: "Line Trace",
        },
        {
          value: "cartoon",
          name: "Cartoon",
        },
        {
          value: "tube",
          name: "Tube",
        },
        {
          value: "spheres",
          name: "Spheres",
        },
        {
          value: "ballsAndSticks",
          name: "Balls and Sticks",
        },
      ],

      columns: [
        {
          title: "Name",
          dataIndex: "name",
          key: "name",
        },
        {
          title: "Path",
          dataIndex: "path",
          key: "path",
        },
        {
          title: "Color",
          dataIndex: "color",
          key: "color",
        },
        {
          title: "Action",
          key: "action",
        },
      ],
    };
  },
  mounted() {
    this.create_view();
    if (this.pdb_list) {
      forEach(this.pdb_list, (pdb) => {
        this.add_pdb(pdb);
      });
    } else if (this.pdb) {
      this.add_pdb(this.pdb);
    }
    // this.plot_heatmap();
  },
  methods: {
    create_view: function () {
      var options = {
        width: this.width,
        height: this.height,
        antialias: true,
        quality: "high",
      };
      // clear #protein_viewer
      document.getElementById("protein_viewer").innerHTML = "";
      // insert the viewer under the Dom element with id 'gl'.
      this.viewer = pv.Viewer(
        document.getElementById("protein_viewer"),
        options
      );
      this.bind_highlight();
      this.bind_mouse_watching();
    },

    add_pdb: function (pdb) {
      pdb.key = pdb.pdb_path;
      if (this.pdbs.find((p) => p.key === pdb.key)) {
        message.info("PDB already added");
        return;
      }
      pv.io.fetchPdb(API.pdb_url(pdb.pdb_path), (structure) => {
        if (structure) {
          pdb.color = randomColor({ luminosity: "light" });
          pdb.path = split(pdb.pdb_path, "/").pop();

          pv.mol.assignHelixSheet(structure);
          pdb.structure = structure;
          pdb.origin_structure = cloneDeep(structure);
          this.render_structure(pdb);

          this.pdbs.push(pdb);
          if (pdb.visible != null && !pdb.visible) {
            this.viewer.hide(pdb.key);
          } else {
            this.viewing_pdb_keys.push(pdb.key);
          }
        }
      });
    },

    render_structure: function (pdb) {
      this.rendering = true;
      // this.align_structure(pdb);
      this.viewer.renderAs(pdb.key, pdb.structure, this.mode, {
        color: pv.color.uniform(pdb.color),
      });
      this.viewer.autoZoom();
      this.rendering = false;
    },

    align_structure: function (pdb) {
      if (this.viewing_pdb_keys.length > 1) {
        let reference = this.pdbs.find(
          (p) => p.key === this.viewing_pdb_keys[0]
        ).structure;
        pv.mol.superpose(pdb.structure, reference);
      }
    },

    align_all: function () {
      if (this.viewing_pdb_keys.length >= 2) {
        let reference = this.pdbs.find(
          (p) => p.key === this.viewing_pdb_keys[0]
        ).structure;
        for (let i = 1; i < this.viewing_pdb_keys.length; i++) {
          let pdb = this.pdbs.find((p) => p.key === this.viewing_pdb_keys[i]);
          pv.mol.superpose(pdb.structure, reference);
        }
        this.rerender();
      }
    },

    tmalign_all: function () {
      if (this.viewing_pdb_keys.length >= 2) {
        let pdbs_path = [];
        let pdbs = [];
        for (let i = 0; i < this.viewing_pdb_keys.length; i++) {
          let pdb = this.pdbs.find((p) => p.key === this.viewing_pdb_keys[i]);
          pdbs_path.push(pdb.pdb_path);
          pdbs.push(pdb);
        }
        this.tm_aligning = true;
        axios(API.align_pdbs(pdbs_path))
          .then((res) => {
            console.log(res.data);
            this.plot_heatmap(pdbs, res.data);
            this.transform_structures(pdbs, res.data.rotation);
          })
          .catch((err) => {
            console.log(err);
          })
          .finally(() => {
            this.tm_aligning = false;
          });
      }
    },

    transform_structures: function (pdbs, m_transform) {
      let reference_pdb = pdbs[pdbs.length - 1];
      // reset the structure of reference_pdb from origin_structure
      reference_pdb.structure = cloneDeep(reference_pdb.origin_structure);

      for (let i = 0; i < pdbs.length - 1; i++) {
        let pdb = pdbs[i];
        let translation = map(m_transform[i][pdbs.length - 1], (v) => v[0]);
        let rotation = map(m_transform[i][pdbs.length - 1], (v) => v.slice(1));
        console.log("translation", translation);
        console.log("rotation", rotation);
        let origin_atoms = pdb.origin_structure.full().atoms();
        let atoms = pdb.structure.full().atoms();
        for (let j = 0; j < origin_atoms.length; j++) {
          let atom = origin_atoms[j];
          let new_atom = atoms[j];
          let x = atom.pos()[0];
          let y = atom.pos()[1];
          let z = atom.pos()[2];
          let new_x =
            x * rotation[0][0] +
            y * rotation[0][1] +
            z * rotation[0][2] +
            translation[0];
          let new_y =
            x * rotation[1][0] +
            y * rotation[1][1] +
            z * rotation[1][2] +
            translation[1];
          let new_z =
            x * rotation[2][0] +
            y * rotation[2][1] +
            z * rotation[2][2] +
            translation[2];
          new_atom.setPos([new_x, new_y, new_z]);
        }
      }
      this.rerender();
    },

    plot_heatmap: function (pdbs, result) {
      Plotly.newPlot(
        this.$refs.heatmap_tmscore,
        [
          {
            type: "heatmap",
            x: pdbs.map((p) => p.name),
            y: pdbs.map((p) => p.name),
            z: result.tm_score,
            zmin: 0,
            zmax: 1,
            colorscale: "YlGnBu",
          },
        ],
        {
          title: {
            text: "TM-score",
            font: {
              family: "Arial",
              size: 18,
              color: "#000000",
            },
          },
          annotations: this.get_annotation(
            pdbs.map((p) => p.name),
            pdbs.map((p) => p.name),
            result.tm_score,
            (z) => {
              if (z < 0.8) {
                return "white";
              } else {
                return "black";
              }
            }
          ),
        }
      );
      Plotly.newPlot(
        this.$refs.heatmap_rmsd,
        [
          {
            type: "heatmap",
            x: pdbs.map((p) => p.name),
            y: pdbs.map((p) => p.name),
            z: result.rmsd,
            colorscale: "YlGnBu",
            reversescale: true,
          },
        ],
        {
          title: {
            text: "RMSD",
            font: {
              family: "Arial",
              size: 18,
              color: "#000000",
            },
          },
          annotations: this.get_annotation(
            pdbs.map((p) => p.name),
            pdbs.map((p) => p.name),
            result.rmsd,
            (z) => {
              if (z > 0.5) {
                return "white";
              } else {
                return "black";
              }
            }
          ),
        }
      );
    },

    get_annotation: function (xValues, yValues, zValues, color_func) {
      let annotations = [];
      for (var i = 0; i < yValues.length; i++) {
        for (var j = 0; j < xValues.length; j++) {
          var currentValue = zValues[i][j];
          var textColor = "black";
          if (color_func) {
            textColor = color_func(currentValue);
          }
          var result = {
            xref: "x1",
            yref: "y1",
            x: xValues[j],
            y: yValues[i],
            text: zValues[i][j],
            font: {
              family: "Arial",
              size: 12,
              color: "rgb(50, 171, 96)",
            },
            showarrow: false,
            font: {
              color: textColor,
            },
          };
          annotations.push(result);
        }
      }
      return annotations;
    },

    remove_pdb: function (pdb) {
      this.pdbs = this.pdbs.filter((item) => item.key !== pdb.key);
      this.viewing_pdb_keys = this.viewing_pdb_keys.filter(
        (key) => key !== pdb.key
      );
      this.viewer.hide(pdb.key);
      this.viewer.rm(pdb.key);
      if (this.viewing_pdb_keys.length > 0) {
        this.viewer.autoZoom();
      }
    },

    download_pdb: function (pdb) {
      var a = document.createElement("a");
      a.target = "_blank";
      a.href = API.download_url(pdb.pdb_path);
      a.click();
    },

    setColorForAtom: function (go, atom, color) {
      var view = go.structure().createEmptyView();
      view.addAtom(atom);
      go.colorBy(pv.color.uniform(color), view);
    },

    clear_all: function () {
      forEach(this.pdbs, (pdb) => {
        this.remove_pdb(pdb);
      });
      this.viewing_pdb_keys = [];
      this.pdbs = [];
    },

    on_select_change: function (pdb_paths) {
      forEach(this.pdbs, (pdb) => {
        if (pdb_paths.includes(pdb.pdb_path)) {
          this.viewer.show(pdb.pdb_path);
          this.viewer.autoZoom();
        } else {
          this.viewer.hide(pdb.pdb_path);
          this.viewer.autoZoom();
        }
      });
      this.viewing_pdb_keys = pdb_paths;
    },

    change_color: function (record) {
      record.color = randomColor({ luminosity: "light" });
      this.viewer.get(record.key).colorBy(pv.color.uniform(record.color));
      this.viewer.autoZoom();
    },

    rerender: function () {
      this.viewer.clear();
      forEach(this.viewing_pdb_keys, (key) => {
        let pdb = this.pdbs.find((p) => p.key === key);
        this.render_structure(pdb);
      });
    },

    bind_mouse_watching: function () {
      document
        .getElementById("protein_viewer")
        .addEventListener("mousedown", (e) => {
          this.dragging = true;
        });

      document
        .getElementById("protein_viewer")
        .addEventListener("mouseup", (e) => {
          this.dragging = false;
        });
    },

    bind_highlight: function () {
      document
        .getElementById("protein_viewer")
        .addEventListener("mousemove", (event) => {
          if (!this.dragging && this.show_residue) {
            var rect = this.viewer.boundingClientRect();
            var picked = this.viewer.pick({
              x: event.clientX - rect.left,
              y: event.clientY - rect.top,
            });
            if (
              this.prevPicked !== null &&
              picked !== null &&
              picked.target() === this.prevPicked.atom
            ) {
              return;
            }
            if (this.prevPicked !== null) {
              // reset color of previously picked atom.
              this.setColorForAtom(
                this.prevPicked.node,
                this.prevPicked.atom,
                this.prevPicked.color
              );
            }
            if (picked !== null) {
              var atom = picked.target();
              document.getElementById("picked-atom-name").innerHTML =
                atom.qualifiedName();
              // get RGBA color and store in the color array, so we know what it was
              // before changing it to the highlight color.
              var color = [0, 0, 0, 0];
              picked.node().getColorForAtom(atom, color);
              this.prevPicked = {
                atom: atom,
                color: color,
                node: picked.node(),
              };

              this.setColorForAtom(picked.node(), atom, "red");
            } else {
              document.getElementById("picked-atom-name").innerHTML = "&nbsp;";
              this.prevPicked = null;
            }
            this.viewer.requestRedraw();
          }
        });
    },
  },
  watch: {
    pdb: function (new_pdb) {
      this.add_pdb(new_pdb);
    },

    pdb_list: function (new_pdb_list) {
      this.clear_all();
      forEach(new_pdb_list, (pdb) => {
        this.add_pdb(pdb);
      });
    },

    mode: function (new_mode) {
      this.rerender();
    },
  },
  computed: {},
};
</script>
