var FlipperView = Backbone.View.extend({
    flipId:  '#flip',

    initialize: function(argc) {
        this.heads               = argc.heads;
        this.tails               = argc.tails;
        this.triggerName         = argc.triggerName;
        this.initializeAnimation = typeof argc.initializeAnimation === 'undefined' ?  true : false;
        this.template            = _.template($(argc.template).html()),

        this.current     = null;

        Backbone.on(this.triggerName, this.flip, this);
        var $this = this;
        Backbone.on(`${this.triggerName}.animate`, function(x) {
            $this.runAnimation(x);
        }, this);
    },

    render: function() {
        this.$el.html(
            this.template()
        );

        this.flip();

        return this;
    },

    initializeFlip: function() {
        this.current = this.heads;
        this.show(this.current, this.initializeAnimation);
    },

    show: function(node, withAnimation) {
        if (node.$el) {
            var el = this.$(this.flipId);

            node.setElement(el);
        }

        node.render();

        withAnimation &&
            node.animationShow();
    },

    flip: function() {

        if (!this.current) {
            this.initializeFlip();
            return;
        }
        
        var next = this.current == this.tails ? this.heads : this.tails;
        
        var $this = this;

        this.current.animationHide(function() {
            $this.show(next, true);
        });
        
        this.current = next;
    }
    
});

_.extend(FlipperView.prototype, AnimatableMixin);