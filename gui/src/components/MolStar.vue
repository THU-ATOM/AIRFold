<template>
  <div>
    <div class="w-full">
      <div id="molstar_viewer" class="relative w-full h-96 z-50"></div>
    </div>
  </div>
</template>

<script>
import { API } from "@/js/api";

export default {
  name: "MolStarViewer",
  props: {
    pdb: Object,
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

      visual_style: "cartoon",
      visual_styles: [
        {
          value: "cartoon",
          name: "Cartoon",
        },
        {
          value: "ball-and-stick",
          name: "Balls and Sticks",
        },
        {
          value: "carbohydrate",
          name: "Carbohydrate",
        },
        {
          value: "distance-restraint",
          name: "Distance Restraint",
        },
        {
          value: "ellipsoid",
          name: "Ellipsoid",
        },
        {
          value: "gaussian-surface",
          name: "Gaussian Surface",
        },
        {
          value: "molecular-surface",
          name: "Molecular Surface",
        },
        {
          value: "point",
          name: "Point",
        },
        {
          value: "putty",
          name: "Putty",
        },
        {
          value: "spacefill",
          name: "Spacefill",
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
    // if (this.pdb_list) {
    //   forEach(this.pdb_list, (pdb) => {
    //     this.add_pdb(pdb);
    //   });
    // } else if (this.pdb) {
    //   this.add_pdb(this.pdb);
    // }
    // this.plot_heatmap();
  },
  computed: {
    default_options() {
      return {
        visualStyle: this.visual_style,
        bgColor: { r: 255, g: 255, b: 255 },
        // alphafoldView: true,
        moleculeId: "4hr2",
        // customData: {
        //   url: API.pdb_url("/data/protein/CASP15/data/2022-06-24/structure/intergrated_fa/2022-06-24_T1158v3___ppdI/rank_1_model_3_seed_0_relaxed.cif"),
        //   format: "cif",
        // },
        domainAnnotation: true,
        hideControls: true,
      };
    },
  },
  methods: {
    create_view: function () {
      // var instance = new PDBeMolstarPlugin();
      // instance.render(
      //   document.getElementById("protein_viewer"),
      //   this.default_options
      // );
      // this.viewer = instance;
      // instance.visual.update({moleculeId: '4dut'}, false);

      //Create plugin instance
      var viewerInstance = new PDBeMolstarPlugin();
      this.viewer = viewerInstance;

      //Set options (Checkout available options list in the documentation)
      // var options = {
      //   moleculeId: '4hr2',
      //   // hideControls: true
      // }

      //Get element from HTML/Template to place the viewer
      var viewerContainer = document.getElementById("molstar_viewer");

      //Call render method to display the 3D view
      viewerInstance.render(viewerContainer, this.default_options);

      // Load second model
      // viewerInstance.visual.update({moleculeId: '4dut'}, false);
    },

    add_pdb: function (pdb) {
      this.pdbs.push(pdb);
    },

    show_pdb: function (pdb, append) {
        this.viewer.visual.update(
          {
            ...this.default_options,
            customData: {
              url: API.pdb_url(pdb.pdb_path),
              format: "pdb",
            },
          },
          !append
        );
    },
  },
  watch: {
    pdb: function (new_pdb) {
      this.add_pdb(new_pdb);
      this.show_pdb(new_pdb, false);
    },
  },
};
</script>

<style>
</style>