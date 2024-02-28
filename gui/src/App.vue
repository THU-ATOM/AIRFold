<template>
  <a-layout class="layout">
    <a-layout-header>
      <a-menu
        v-model:selectedKeys="selectedKeys"
        theme="dark"
        mode="horizontal"
        :style="{ lineHeight: '64px' }"
        class="flex"
      >
        <a-menu-item @click="$router.push('/')" key="score"
          >ScoreBoard</a-menu-item
        >
        <a-menu-item @click="$router.push('/analyze')" key="analyze"
          >Analysis</a-menu-item
        >
        <a-menu-item @click="$router.push('/compare')" key="compare"
          >Comparison</a-menu-item
        >
        <!-- <a-menu-item key="2">CASP15</a-menu-item>
        <a-menu-item key="3">nav 3</a-menu-item> -->
      </a-menu>
    </a-layout-header>
    <a-layout-content style="padding: 0 50px">
      <div class="flex justify-between items-center">
        <a-breadcrumb class="my-4">
          <a-breadcrumb-item>Home</a-breadcrumb-item>
          <a-breadcrumb-item v-if="$route.path == '/'"
            >Scoreboard</a-breadcrumb-item
          >
          <a-breadcrumb-item v-if="$route.path == '/analyze'"
            >Analysis</a-breadcrumb-item
          >
          <a-breadcrumb-item v-if="$route.path == '/compare'"
            >Comparison</a-breadcrumb-item
          >
        </a-breadcrumb>
        <div class="flex pr-5 text-base">
          <a-dropdown class="ml-4">
            <link-outlined />
            <template #overlay>
              <a-menu>
                <a-menu-item v-for="link in links" :key="link.name">
                  <a :href="link.href" target="_blank">{{ link.name }}</a>
                </a-menu-item>
              </a-menu>
            </template>
          </a-dropdown>
          <a-dropdown class="ml-4">
            <tool-outlined />
            <template #overlay>
              <a-menu>
                <a-menu-item v-for="tool in tools" :key="tool.name">
                  <a :href="tool.href" target="_blank">{{ tool.name }}</a>
                </a-menu-item>
              </a-menu>
            </template>
          </a-dropdown>
        </div>
      </div>
      <div :style="{ background: '#fff', padding: '24px', minHeight: '280px' }">
        <router-view />
      </div>
    </a-layout-content>
    <a-layout-footer style="text-align: center">
      Air-Health ©2022
    </a-layout-footer>
  </a-layout>
</template>

<script>
import { LinkOutlined, ToolOutlined } from "@ant-design/icons-vue";

export default {
  mounted: function () {},
  data() {
    return {
      selectedKeys: ["score"],
      links: [
        {
          name: "CASP15 Target List",
          href: "https://predictioncenter.org/casp15/targetlist.cgi",
        },
        {
          name: "CAOME 3D 1-week Performance",
          href: "https://www.cameo3d.org/modeling/1-week/difficulty/all",
        },
      ],
      tools: [
        {
          name: "Protein BLAST",
          href: "https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastp&PAGE_TYPE=BlastSearch&LINK_LOC=blasthome",
        },
        {
          name: "UniprotKB",
          href: "https://www.uniprot.org/",
        },
        {
          name: "PubMed",
          href: "https://pubmed.ncbi.nlm.nih.gov/",
        },
      ],
    };
  },
  watch: {
    $route: function (new_route) {
      if (new_route.path == "/") {
        this.selectedKeys = ["score"];
      } else if (new_route.path == "/analyze") {
        this.selectedKeys = ["analyze"];
      } else if (new_route.path == "/compare") {
        this.selectedKeys = ["compare"];
      }
    },
  },
  components: {
    LinkOutlined,
    ToolOutlined,
  },
};
</script>
<style>
.site-layout-content {
  min-height: 280px;
  padding: 24px;
  background: #fff;
}

[data-theme="dark"] .site-layout-content {
  background: #141414;
}
</style>
